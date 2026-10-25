from utils import *
from evaluation_kit import *
from simpleqa_helpers import grade_one_sample


def add_data_config_and_postprocess(parser):
    args = parser.parse_args()

    args.benchmark = "deepsearchqa"
    args.sample_id_key = "uid"
    args.question_key = "question"
    # args.reference_key = "urls" # no reference URL
    args.available_retrievers = [WebRetriever()]
    return args


def load_benchmark(args):
    dataset = load_dataset("google/deepsearchqa")["eval"]
    dataset = dataset.filter(lambda x: x['answer_type'] == "Single Answer")
    dataset = dataset.select(range(200))
    if args.phase == "benchmark_probing":
        dataset = dataset.select(range(50))
    dataset = dataset.map(
        lambda x: {**x, "question": x['problem'], "uid": x['problem'][:50].lower().replace(" ", "-")},
        desc="Add unique ID column",
    )
    return dataset


if __name__ == "__main__":
    evaluation(
        add_data_config_and_postprocess=add_data_config_and_postprocess,
        load_benchmark=load_benchmark,
        grade_one_sample=grade_one_sample,
    )
