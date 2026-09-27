import pytest

from data.self_distill import (
    arithmetic_reward,
    build_arithmetic_tasks,
    build_prompt_dataset,
    collect_rollouts,
    verify_completion,
)


class TestBuildArithmeticTasks:
    def test_deterministic_across_calls(self):
        assert build_arithmetic_tasks(50, seed=7) == build_arithmetic_tasks(50, seed=7)

    def test_split_assigned_before_collection(self):
        tasks = build_arithmetic_tasks(100, seed=42, eval_fraction=0.2)
        assert sum(task["split"] == "eval" for task in tasks) == 20
        assert sum(task["split"] == "train" for task in tasks) == 80

    def test_unique_questions_and_correct_answers(self):
        tasks = build_arithmetic_tasks(200, seed=1)
        assert len({task["question"] for task in tasks}) == 200
        for task in tasks[:20]:
            a, b = (int(n) for n in task["question"].split("?")[0].split() if n.isdigit())
            assert task["answer"] == a + b


class TestVerifyCompletion:
    @pytest.mark.parametrize(
        ("completion", "answer", "correct"),
        [
            ("85", 85, True),
            ("The answer is 85.", 85, True),
            ("47 + 38 = 85", 85, True),  # last integer wins
            ("84", 85, False),
            ("no number here", 85, False),
            ("85 is wrong, it is 86", 85, False),
        ],
    )
    def test_parses_last_integer(self, completion, answer, correct):
        assert verify_completion(completion, answer)["correct"] is correct

    def test_reports_parsed_value(self):
        assert verify_completion("it is 12", 13) == {"correct": False, "parsed": 12}
        assert verify_completion("???", 13) == {"correct": False, "parsed": None}


class TestCollectRollouts:
    def test_refuses_eval_split_tasks(self):
        tasks = [{"task_id": "t", "split": "eval", "question": "q", "answer": 1}]
        with pytest.raises(ValueError, match="eval tasks never yield training traces"):
            collect_rollouts(object(), object(), tasks)


class TestBuildPromptDataset:
    def test_conversational_prompt_and_reward_columns(self):
        ds = build_prompt_dataset("train", n_tasks=50)
        assert set(ds.column_names) == {"prompt", "answer", "privileged_context"}
        row = ds[0]
        assert row["prompt"][0]["role"] == "user"
        assert str(row["answer"]) in row["privileged_context"]

    def test_eval_never_overlaps_train(self):
        train = {r[0]["content"] for r in build_prompt_dataset("train", n_tasks=50)["prompt"]}
        held_out = {r[0]["content"] for r in build_prompt_dataset("eval", n_tasks=50)["prompt"]}
        assert train and held_out and not train & held_out

    def test_bad_split_raises(self):
        with pytest.raises(ValueError, match="split"):
            build_prompt_dataset("test", n_tasks=10)


class TestArithmeticReward:
    def test_scores_standard_and_conversational_completions(self):
        completions = ["12 + 7 = 19", [{"role": "assistant", "content": "19"}], "20"]
        assert arithmetic_reward(completions, answer=[19, 19, 19]) == [1.0, 1.0, 0.0]

    def test_malformed_completions_score_the_floor(self):
        assert arithmetic_reward(["", [], [{"no": "content"}]], answer=[1, 1, 1]) == [0.0, 0.0, 0.0]
