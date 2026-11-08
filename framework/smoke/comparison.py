# ABOUTME: Compares scientific results and optimizer trajectories of representative smoke runs.
# ABOUTME: Identifies the first divergence and largest endpoint changes at saved precision.
import argparse
import csv
import json
import math
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from itertools import islice, zip_longest
from pathlib import Path

CASE_FIELDS = ("molecule", "algorithm", "target_n_params", "noise_model", "noise_strength")
METADATA = {"runtime_seconds", "timestamp", "output_dir", "output_root"}


class Differences:
    def __init__(self):
        self.first = None
        self.trajectory = None
        self.input_difference = None
        self.largest = {}

    def record(self, where, expected, actual, *, metric=None, trajectory=False):
        if expected == actual:
            return
        message = f"{where}: {expected!r} != {actual!r}"
        if self.first is None:
            self.first = message
        if trajectory and self.trajectory is None:
            self.trajectory = message
        if metric:
            try:
                delta = abs(Decimal(str(actual)) - Decimal(str(expected)))
            except (InvalidOperation, TypeError, ValueError):
                return
            self.measure(metric, delta, where)

    def measure(self, metric, delta, where):
        if delta.is_finite() and (metric not in self.largest or delta > self.largest[metric][0]):
            self.largest[metric] = (delta, where)

    def check(self):
        if self.first:
            lines = [self.first]
            if self.trajectory and self.trajectory != self.first:
                lines.append(f"First differing trajectory record: {self.trajectory}")
            if self.input_difference:
                lines.append(f"First differing optimizer input: {self.input_difference}")
            for metric, (delta, where) in self.largest.items():
                lines.append(f"Largest {metric} difference {delta:.12g} at {where}")
            raise ValueError("\n".join(lines))


def _table(path):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = [field for field in reader.fieldnames or [] if field not in METADATA]
        if not fields or len(fields) != len(set(fields)):
            raise ValueError(f"{path}: missing or duplicate column names")
        cases = defaultdict(list)
        for row in reader:
            if None in row or any(row[field] is None for field in fields):
                raise ValueError(f"{path}: malformed row at line {reader.line_num}")
            key = tuple(f"{field}={row[field]}" for field in CASE_FIELDS if field in fields)
            cases[key].append({field: row[field] for field in fields})
    if not cases:
        raise ValueError(f"{path}: no scientific rows")
    return set(fields), cases


def _compare_table(left, right, name, diff):
    fields, a = _table(left / name)
    other_fields, b = _table(right / name)
    diff.record(f"{name} columns", sorted(fields), sorted(other_fields))
    diff.record(f"{name} cases", sorted(a), sorted(b))
    trace = name.endswith("_traces.csv")
    for case in a:
        if case not in b:
            continue
        rows = a[case]
        other = b[case]
        label = f"{name} {' '.join(case)}"
        diff.record(f"{label} row count", len(rows), len(other), trajectory=trace)
        for index, (x, y) in enumerate(zip(rows, other)):
            position = f"row {index + 1}"
            for identity in ("iteration", "eval_index"):
                if identity in x:
                    position = f"{identity}={x[identity]}"
            for field in x:
                if field not in y:
                    continue
                metric = None
                if field == "final_energy" and x.get("algorithm") == "hardware_efficient_vqe":
                    metric = "HE final energy (Ha)"
                elif field in ("energy", "best_error_vs_casci"):
                    metric = "trace energy (Ha)"
                elif field in ("energy_noisy", "energy_clean_at_noisy_params"):
                    metric = "noise endpoint energy (Ha)"
                diff.record(f"{label} {position} {field}", x[field], y[field],
                            metric=metric, trajectory=trace)
    return sum(map(len, a.values()))


def _compare_json(x, y, path, diff, *, parameters=False):
    if isinstance(x, dict) and isinstance(y, dict):
        keys = x.keys() - METADATA
        other = y.keys() - METADATA
        diff.record(f"{path} fields", sorted(keys), sorted(other))
        for key in sorted(keys & other):
            _compare_json(x[key], y[key], f"{path}/{key}", diff, parameters=parameters)
    elif isinstance(x, list) and isinstance(y, list):
        diff.record(f"{path} length", len(x), len(y))
        for index, (a, b) in enumerate(zip(x, y)):
            _compare_json(a, b, f"{path}[{index}]", diff, parameters=parameters)
    else:
        diff.record(path, x, y, metric="noise parameter (rad)" if parameters else None)


def _event_cases(stream):
    positions = {}
    offset = 0
    previous = None
    for line_number, raw in enumerate(stream, 1):
        try:
            event = json.loads(raw)
            workload = event["workload"]
            if workload not in ("amino", "hydrogen", "noise"):
                raise ValueError(f"unexpected workload {workload!r}")
            fields = ("workload", "molecule", "algorithm")
            if not all(isinstance(event[f], str) and event[f] for f in fields):
                raise ValueError("expected nonempty workload, molecule and algorithm")
            if workload == "noise":
                fields += ("noise_model", "strength")
                if (event["noise_model"] is not None and not isinstance(event["noise_model"], str)
                        or type(event["strength"]) not in (float, int) or not math.isfinite(event["strength"])):
                    raise ValueError("invalid noise_model or strength")
            else:
                fields += ("target",)
                if type(event["target"]) is not int or event["target"] < 1:
                    raise ValueError("invalid target")
            case = tuple((field, event[field]) for field in fields)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{stream.name} line {line_number}: invalid case: {exc}") from exc
        try:
            if (type(event["callback"]) is not bool or not isinstance(event["operators"], list)
                    or any(type(i) is not int or i < 0 for i in event["operators"])
                    or len(event["parameters"]) != 64 or len(bytes.fromhex(event["parameters"])) != 32
                    or not math.isfinite(float.fromhex(event["energy"]))):
                raise ValueError("invalid callback, operators, parameters or energy")
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{stream.name} line {line_number}: malformed event for {dict(case)}: {exc}") from exc
        if case != previous:
            if case in positions:
                raise ValueError(f"{stream.name} line {line_number}: noncontiguous case {dict(case)}")
            positions[case] = [offset, 0]
            previous = case
        positions[case][1] += 1
        offset += len(raw)
    if not positions:
        raise ValueError(f"{stream.name}: empty diagnostic")
    return positions


def _case_events(stream, position):
    offset, count = position
    stream.seek(offset)
    for raw in islice(stream, count):
        yield json.loads(raw)


def _input_key(event):
    return event["callback"], event["parameters"], tuple(event["operators"])


def _compare_case(ref, cand, ref_position, cand_position, case, diff):
    label = " ".join(f"{k}={v}" for k, v in case)
    expected = list(_case_events(ref, ref_position))
    # Extremes cover repeated requests for the same input, including unequal multiplicities.
    ranges = {}
    for index, event in enumerate(expected, 1):
        key = _input_key(event)
        sample = (float.fromhex(event["energy"]), index)
        low, high = ranges.get(key, (sample, sample))
        ranges[key] = min(low, sample), max(high, sample)
    for index, (x, y) in enumerate(zip_longest(expected, _case_events(cand, cand_position)), 1):
        where = f"events.jsonl event {index} {label}"
        if x != y:
            if diff.input_difference is None and (x is None or y is None or _input_key(x) != _input_key(y)):
                diff.input_difference = where
            if x is None or y is None:
                diff.record(f"{where} present", x is not None, y is not None, trajectory=True)
            else:
                for field in sorted(x.keys() | y.keys()):
                    if field not in x or field not in y:
                        diff.record(f"{where} field {field}", field in x, field in y, trajectory=True)
                    else:
                        diff.record(f"{where} {field}", x[field], y[field], trajectory=True)
        if y is not None and _input_key(y) in ranges:
            energy = float.fromhex(y["energy"])
            for ref_energy, ref_index in ranges[_input_key(y)]:
                delta = abs(ref_energy - energy)
                if delta:
                    diff.measure("fixed-input event energy (Ha)", Decimal.from_float(delta),
                                 f"{where} energy (reference event {ref_index})")


def _events(left, right, diff):
    a, b = left / "events.jsonl", right / "events.jsonl"
    if not a.exists() and not b.exists():
        return None
    if not (a.is_file() and b.is_file()):
        raise ValueError("Both runs must provide events.jsonl for a binary comparison")
    with a.open("rb") as ref, b.open("rb") as cand:
        reference = _event_cases(ref)
        candidate = _event_cases(cand)
        for position, (x, y) in enumerate(zip_longest(reference, candidate), 1):
            if x != y:
                where = f"events.jsonl case order at case {position}, event 1"
                ref_label, candidate_label = dict(x) if x else None, dict(y) if y else None
                diff.record(where, ref_label, candidate_label, trajectory=True)
                if diff.input_difference is None:
                    diff.input_difference = f"{where}: {ref_label} != {candidate_label}"
            if x in candidate:
                _compare_case(ref, cand, reference[x], candidate[x], x, diff)
        return sum(count for _, count in reference.values())


def _geometry_hashes(directory, workload):
    path = directory / workload / "measurement.json"
    if not path.is_file():
        raise ValueError(f"Missing geometry provenance: {path}")
    measurement = json.loads(path.read_text())
    hashes = measurement.get("geometries") if isinstance(measurement, dict) else None
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError(f"{path}: missing geometry hashes")
    for molecule, digest in hashes.items():
        if (not molecule or not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)):
            raise ValueError(f"{path}: invalid geometry hash for {molecule!r}")
    return hashes


def _compare_suite_inputs(reference, candidate):
    manifests = [root / "suite.json" for root in (reference, candidate)]
    if not any(path.exists() or path.is_symlink() for path in manifests):
        return
    if not all(path.is_file() for path in manifests):
        raise ValueError("Both runs must provide suite.json when comparing saved suites")
    from smoke.representative import WORKLOADS, saved_profile

    expected, actual = saved_profile(reference), saved_profile(candidate)
    if expected != actual:
        raise ValueError(f"Smoke suite scope differs: {expected} != {actual}")
    for workload in WORKLOADS:
        left = _geometry_hashes(reference, workload)
        right = _geometry_hashes(candidate, workload)
        for molecule in sorted(left.keys() | right.keys()):
            if left.get(molecule) != right.get(molecule):
                raise ValueError(f"{workload}/measurement.json geometry {molecule}: "
                                 f"{left.get(molecule)!r} != {right.get(molecule)!r}")


def compare(reference: Path, candidate: Path) -> dict:
    """Require equality of saved scientific artifacts for two completed same-backend runs."""
    reference, candidate = Path(reference), Path(candidate)
    _compare_suite_inputs(reference, candidate)
    diff = Differences()
    tables = []
    configs = ["amino/batch_manifest.json", "hydrogen/run_config.json", "noise/run_config.json",
               "noise/optimum_parameters.json"]
    if not (reference / configs[0]).is_file():
        raise ValueError(f"Missing reference artifact {configs[0]}")
    manifest = json.loads((reference / configs[0]).read_text())
    for root, prefix in [("amino", "all_molecules_"), ("hydrogen", "")]:
        for stem in ("results", "convergence_traces", "eval_traces"):
            tables.append(f"{root}/{prefix}{stem}.csv")
    tables += ["noise/results.csv", "noise/convergence_traces.csv",
               "amino/all_molecules_cost_to_accuracy_summary.csv"]
    for molecule in manifest["molecules_succeeded"]:
        configs.append(f"amino/{molecule}/run_config.json")
        for stem in ("results", "convergence_traces", "eval_traces"):
            tables.append(f"amino/{molecule}/{stem}.csv")
    for name in configs + tables:
        for root in (reference, candidate):
            if not (root / name).is_file():
                raise ValueError(f"Missing scientific artifact {root / name}")
    for name in configs:
        x = json.loads((reference / name).read_text())
        y = json.loads((candidate / name).read_text())
        _compare_json(x, y, name, diff, parameters=name.endswith("optimum_parameters.json"))
    count = sum(_compare_table(reference, candidate, name, diff) for name in tables)
    events = _events(reference, candidate, diff)
    diff.check()
    return {"csv_rows_compared": count, "event_records_compared": events}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    from smoke.representative import validate

    validate(args.reference)
    validate(args.candidate)
    print(json.dumps(compare(args.reference, args.candidate), indent=2))
