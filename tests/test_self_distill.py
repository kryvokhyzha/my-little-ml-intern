import pytest

from data.self_distill import build_arithmetic_tasks, collect_rollouts, verify_completion


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
