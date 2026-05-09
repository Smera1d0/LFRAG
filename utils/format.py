import argparse
import json
from pathlib import Path


DEFAULT_INPUT = Path("../datasets/eval/LF_Docmatix.jsonl")
DEFAULT_OUTPUT = Path("../datasets/eval/LF_Docmatix_formatted.jsonl")


def transform_record(record):
    base_record = {key: value for key, value in record.items() if key != "QAs"}
    qa_items = record.get("QAs", [])

    if not qa_items:
        return [base_record]

    formatted_records = []
    for qa in qa_items:
        formatted_record = dict(base_record)
        formatted_record["question"] = qa.get("question")
        formatted_record["answer"] = qa.get("answer")
        formatted_record["relevant_bbox_ids"] = qa.get("relevant_bbox_ids", [])
        formatted_records.append(formatted_record)
    return formatted_records


def convert_jsonl(input_path, output_path):
    total_input = 0
    total_output = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as target:
        for line in source:
            line = line.strip()
            if not line:
                continue

            total_input += 1
            record = json.loads(line)
            for formatted_record in transform_record(record):
                target.write(json.dumps(formatted_record, ensure_ascii=False) + "\n")
                total_output += 1

    return total_input, total_output


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert QAs list in Docmatix JSONL to flat field format."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input JSONL path, default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSONL path, default: {DEFAULT_OUTPUT}",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    total_input, total_output = convert_jsonl(args.input, args.output)
    print(f"Input records: {total_input}")
    print(f"Output records: {total_output}")
    print(f"Output file: {args.output}")


if __name__ == "__main__":
    main()