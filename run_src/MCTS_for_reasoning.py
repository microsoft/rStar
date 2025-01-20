# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import sys

sys.path.append(".")

import numpy as np, os, random, json, math, wandb
from tqdm import trange
from typing import List, Dict, Tuple
from copy import deepcopy

try:
    from rapidfuzz import fuzz, process
except:
    pass

from models.IO_System import IO_System
from common.utils import read_txt, read_json
from eval_src.Evaluator import Evaluator, GSM8KEvaluator
from MCTS_backbone import MCTS_Searcher, MCTS_Node
from run_src.rstar_utils import (
    Node_Type,
    GeneratorError,
    reach_terminal_subquestion,
    reach_terminal_ost_step,
    concat_subqs_and_subas,
    concat_ost_steps,
    concat_subqs_subas_as_ost_steps,
    make_hint,
    make_response_prefix,
    split_user_question,
    print_tree_from_root,
    find_valid_solution_nodes,
    find_best_solution,
    stochastic_find_best_solution,
)


def verbose_print(s: str, verbose: bool):
    if verbose:
        print(s)


class Generator:
    """Generator generates children nodes"""

    def __init__(self, args, tokenizer, model, evaluator: Evaluator) -> None:
        self.io = IO_System(args, tokenizer, model)
        self.evaluator = evaluator

        self.num_subquestions = args.num_subquestions
        self.num_a1_steps = args.num_a1_steps
        self.num_votes = args.num_votes
        self.max_tokens = args.max_tokens
        self.enable_potential_score = args.enable_potential_score

        self.mcts_num_last_votes = args.mcts_num_last_votes

        with open(args.decompose_template_path, "r") as f:
            decompose_template = json.load(f)
            self.question_index = decompose_template["index"]

        self.decompose_prompt = read_txt(args.decompose_prompt_path)
        self.fewshot_cot_prompt = read_txt(args.fewshot_cot_prompt_path)
        self.fewshot_cot_config = read_json(args.fewshot_cot_config_path)

        if not args.disable_a1:  # A1: Propose an one-step thought.
            self.fewshot_ost_prompt = read_txt(args.fewshot_ost_prompt_path)
            self.fewshot_ost_config = read_json(args.fewshot_ost_config_path)

        if not args.disable_a5:  # A5: Rephrase the question/sub-question.
            self.rephrasing_prompt_template = read_txt(args.rephrasing_prompt_template_path)
            self.decompose_prompt_rephrased = read_txt(args.decompose_prompt_rephrased_path)
            self.fewshot_cot_prompt_rephrased = read_txt(args.fewshot_cot_prompt_rephrased_path)
            self.fewshot_ost_prompt_rephrased = read_txt(args.fewshot_ost_prompt_rephrased_path)

    def _extract_from_cache(self, subquestion_list: List[str]):
        high_score_questions = []
        selected_answers = []
        values = []
        low_score_questions = []
        low_score_values = []
        low_score_answers_list = []
        unmatched_questions = []

        for subquestion in subquestion_list:
            best_match = process.extractOne(subquestion, self.reasoning_cache.keys(), scorer=fuzz.ratio)

            if best_match:
                best_question, best_score = best_match[0], best_match[1]
                similarity = best_score / 100
                cache_entry = self.reasoning_cache[best_question]
                score = cache_entry["score"]
                if similarity == 1:
                    if score >= 0.9:
                        high_score_questions.append(best_question)
                        selected_answers.append(cache_entry["selected_answer"])
                        values.append(score)
                    else:
                        low_score_questions.append(best_question)
                        low_score_values.append(score)
                        low_score_answers_list.append(cache_entry["answer_list"])
                else:
                    unmatched_questions.append(subquestion)
            else:
                unmatched_questions.append(subquestion)

        return {
            "high_score_questions": high_score_questions,
            "selected_answers": selected_answers,  # most likely answer corresponding to each subquestion
            "values": values,
            "low_score_questions": low_score_questions,
            "low_score_values": low_score_values,
            "low_score_answers_list": low_score_answers_list,
            "unmatched_questions": unmatched_questions,
        }

    def _get_most_likely_answer(self, io_output_list: List[str]) -> Tuple[str, float]:
        assert len(io_output_list) > 0

        if len(io_output_list) == 1:
            most_confident_answer_full_completion = io_output_list[0]
            confidence = 1
        else:
            _, most_confident_answer_full_completion, _, confidence = self.evaluator.find_most_confident_answer(
                io_output_list
            )
            assert confidence > 0

        return most_confident_answer_full_completion, confidence

    def _fewshot_cot_answer_question(self, question: str, paraphrased: bool, num_return: int, hint: str = None):
        fewshot_cot_prompt = self.fewshot_cot_prompt if not paraphrased else self.fewshot_cot_prompt_rephrased
        question += "\n\n" + hint if hint is not None else ""
        io_input = self.fewshot_cot_config["prompt_template"].format(examples=fewshot_cot_prompt, instruction=question)
        io_output_list = self.io.generate(
            io_input,
            num_return=num_return,
            max_tokens=self.max_tokens,
            stop_tokens=self.fewshot_cot_config["stop_tokens"],
        )
        cleaned_io_output_list = [io_output.strip() for io_output in io_output_list]  #! cleaning
        return io_input, cleaned_io_output_list

    def generate_direct_answers(self, user_question: str, paraphrased: bool, hint: str):
        direct_answer_list, value_list = [], []

        #! few shot cot
        num_return = self.mcts_num_last_votes
        io_input, cleaned_io_output_list = self._fewshot_cot_answer_question(
            question=user_question, paraphrased=paraphrased, num_return=num_return, hint=hint
        )

        try:
            most_likely_answer, likelihood = self._get_most_likely_answer(cleaned_io_output_list)
        except Exception as e:
            raise GeneratorError(
                source="generate direct answer from: few shot cot",
                io_input=io_input,
                io_output_list=cleaned_io_output_list,
            )

        direct_answer_list.append(most_likely_answer)
        value_list.append(likelihood)

        return direct_answer_list, value_list

    def generate_subquestions(
        self,
        user_question: str,
        solution_trace: Dict[int, Dict[str, str]],
        paraphrased: bool,
    ):
        subquestion_list, subanswer_list, value_list = [], [], []
        decompose_prompt = self.decompose_prompt if not paraphrased else self.decompose_prompt_rephrased

        #! generate subquestions
        existing_subquestions_and_subanswers, next_subquestion_id = concat_subqs_and_subas(
            solution_trace, self.question_index
        )
        io_input = (
            decompose_prompt
            + "\n\n"
            + f"Question {self.question_index}: {user_question}"
            + "\n"
            + existing_subquestions_and_subanswers
            + f"Question {self.question_index}.{next_subquestion_id}:"
        )
        io_output_list = self.io.generate(
            io_input,
            max_tokens=128,
            num_return=self.num_subquestions,
            stop_tokens=[
                "\n",
                "\n\n",
                "Answer",
                "Answer ",
                f"Answer {self.question_index}.{next_subquestion_id}",
                f"Answer {self.question_index}.{next_subquestion_id}:",
                f"Answer {self.question_index}.{next_subquestion_id}: ",
            ],
        )

        # subquestion_list = [io_output.split("?")[0] + "?" for io_output in io_output_list]  # cleaning, you might wanna modify this
        subquestion_list = [o.strip() for o in io_output_list]

        #! generate subanswers to the subquestions generated above
        io_input_list = []
        for subquestion in subquestion_list:
            io_input = (
                decompose_prompt
                + "\n\n"
                + f"Question {self.question_index}: {user_question}"
                + "\n"
                + existing_subquestions_and_subanswers
                + f"Question {self.question_index}.{next_subquestion_id}: "
                + subquestion
                + "\n"
                + f"Answer {self.question_index}.{next_subquestion_id}:"
            )
            io_input_list.append(io_input)

        if reach_terminal_subquestion(subquestion=subquestion, user_question=user_question):
            num_return = self.mcts_num_last_votes
        else:
            num_return = self.num_votes

        io_output_list = self.io.generate(
            io_input_list,
            max_tokens=512,
            num_return=num_return,
            stop_tokens=[
                "\n",
                "\n\n",
                f"Question {self.question_index}.{next_subquestion_id + 1}",
            ],
        )
        cleaned_io_output_list = [
            [io_output.strip() for io_output in io_output_group] for io_output_group in io_output_list
        ]

        for i, cleaned_io_output_group in enumerate(cleaned_io_output_list):
            try:
                most_likely_answer, likelihood = self._get_most_likely_answer(cleaned_io_output_group)
            except Exception as e:
                raise GeneratorError(
                    source="generate answer to subquestions",
                    io_input=io_input_list[i],
                    io_output_list=cleaned_io_output_group,
                )
            subanswer_list.append(most_likely_answer)
            value_list.append(likelihood)

        assert len(subquestion_list) == len(subanswer_list) == len(value_list)

        #! generate potential answer to the user question
        potential_answers_list: List[List[str]] = []
        if self.enable_potential_score:
            for subq, suba in zip(subquestion_list, subanswer_list):
                if reach_terminal_subquestion(subq, user_question):
                    potential_answers_list.append(None)
                else:
                    response_prefix = make_response_prefix(
                        solution_trace, Node_Type.SUBQUESTION, new_subq=subq, new_suba=suba
                    )
                    potential_score_input = "Question: " + user_question + "\nAnswer: " + response_prefix

                    potential_score_output = self.io.generate(
                        potential_score_input,
                        num_return=self.num_votes,
                        max_tokens=128,
                        stop_tokens=self.fewshot_cot_config["stop_tokens"],
                    )
                    potential_score_input2 = [
                        "Question: "
                        + user_question
                        + "\nAnswer: "
                        + response_prefix
                        + z
                        + "\nTherefore, the answer (arabic numerals) is"
                        for z in potential_score_output
                    ]
                    cleaned_io_output_list = self.io.generate(
                        potential_score_input2,
                        num_return=1,
                        max_tokens=128,
                        stop_tokens=self.fewshot_cot_config["stop_tokens"],
                    )
                    cleaned_io_output_list = [z[0] for z in cleaned_io_output_list]

                    potential_answers_list.append(
                        [self.evaluator.extract_answer_from_model_completion(o) for o in cleaned_io_output_list]
                    )
        else:
            potential_answers_list = [None] * len(subquestion_list)

        return subquestion_list, subanswer_list, value_list, potential_answers_list

    def generate_re_subanswers(
        self,
        user_question: str,
        solution_trace: Dict[int, Dict[str, str]],
        paraphrased: bool,
    ):
        re_subanswer_list, value_list = [], []

        user_question_context, _ = split_user_question(user_question)

        last_subquestion_id = int(sorted(solution_trace.keys())[-1])
        last_subquestion = solution_trace[last_subquestion_id]["subquestion"]

        #! few shot cot
        question = (
            f"{user_question_context} {last_subquestion}"
            if not paraphrased
            else f"{user_question_context} Question: {last_subquestion}"
        )
        io_input, cleaned_io_output_list = self._fewshot_cot_answer_question(
            question=question, paraphrased=paraphrased, num_return=self.num_votes
        )
        try:
            most_likely_answer, likelihood = self._get_most_likely_answer(cleaned_io_output_list)
        except Exception as e:
            raise GeneratorError(
                source="generate re-subanswers: few shot cot",
                io_input=io_input,
                io_output_list=cleaned_io_output_list,
            )
        re_subanswer_list.append(most_likely_answer)
        value_list.append(likelihood)

        #! generate potential answer to the user question
        potential_answers_list: List[List[str]] = []
        if self.enable_potential_score:
            solution_trace_copy = deepcopy(solution_trace)
            for re_suba in re_subanswer_list:
                solution_trace_copy[last_subquestion_id]["subanswer"] = {"text": re_suba}
                response_prefix = make_response_prefix(solution_trace_copy, Node_Type.SUBQUESTION)
                potential_score_input = "Question: " + user_question + "\nAnswer: " + response_prefix

                potential_score_output = self.io.generate(
                    potential_score_input,
                    num_return=self.num_votes,
                    max_tokens=128,
                    stop_tokens=self.fewshot_cot_config["stop_tokens"],
                )
                potential_score_input2 = [
                    "Question: "
                    + user_question
                    + "\nAnswer: "
                    + response_prefix
                    + z
                    + "\nTherefore, the answer (arabic numerals) is"
                    for z in potential_score_output
                ]
                cleaned_io_output_list = self.io.generate(
                    potential_score_input2,
                    num_return=1,
                    max_tokens=128,
                    stop_tokens=self.fewshot_cot_config["stop_tokens"],
                )
                cleaned_io_output_list = [z[0] for z in cleaned_io_output_list]

                potential_answers_list.append(
                    [self.evaluator.extract_answer_from_model_completion(o) for o in cleaned_io_output_list]
                )
        else:
            potential_answers_list = [None] * len(re_subanswer_list)

        return re_subanswer_list, value_list, potential_answers_list

    def generate_rephrased_user_question(self, user_question: str):
        rephrased_user_question_list = []
        io_input = self.rephrasing_prompt_template
        io_input += "\n\n"
        io_input += "Original Question: " + user_question + "\n"
        io_input += "Rephrased Question: Given a list of conditions, please answer the question. Condition 1: "
        io_output = self.io.generate(model_input=io_input, max_tokens=512, num_return=1, stop_tokens=["\n", "\n\n"])[0]
        io_output = "Given a list of conditions, please answer the question. Condition 1: " + io_output
        rephrased_user_question_list.append(io_output)

        #! generate potential answer to the user question
        potential_answers_list: List[List[str]] = []  # essentially direct answer list
        if self.enable_potential_score:
            response_prefix = make_response_prefix(None, None)
            potential_score_input = "Question: " + rephrased_user_question_list[0] + "\nAnswer: " + response_prefix
            potential_score_output = self.io.generate(
                potential_score_input,
                num_return=self.num_votes,
                max_tokens=128,
                stop_tokens=self.fewshot_cot_config["stop_tokens"],
            )
            potential_score_input2 = [
                "Question: "
                + rephrased_user_question_list[0]
                + "\nAnswer: "
                + response_prefix
                + z
                + "\nTherefore, the answer (arabic numerals) is"
                for z in potential_score_output
            ]
            cleaned_io_output_list = self.io.generate(
                potential_score_input2, num_return=1, max_tokens=128, stop_tokens=self.fewshot_cot_config["stop_tokens"]
            )
            cleaned_io_output_list = [z[0] for z in cleaned_io_output_list]

            potential_answers_list.append(
                [self.evaluator.extract_answer_from_model_completion(o) for o in cleaned_io_output_list]
            )
        else:
            potential_answers_list = [None] * len(rephrased_user_question_list)

        return rephrased_user_question_list, potential_answers_list

    def generate_ost_step(
        self,
        user_question: str,
        solution_trace: Dict[int, Dict[str, str]],
        paraphrased: bool,
        parent_is_subquestion: bool,
    ):
        ost_step_list = []
        if parent_is_subquestion:
            existing_ost_steps, next_ost_step_id = concat_subqs_subas_as_ost_steps(solution_trace)
        else:
            existing_ost_steps, next_ost_step_id = concat_ost_steps(solution_trace)
        io_input = (
            self.fewshot_ost_config["prompt_template"].format(
                examples=self.fewshot_ost_prompt if not paraphrased else self.fewshot_ost_prompt_rephrased,
                instruction=user_question,
            )
            + existing_ost_steps
            + f"Step {next_ost_step_id}:"
        )
        io_output_list = self.io.generate(
            model_input=io_input, max_tokens=256, num_return=self.num_a1_steps, stop_tokens=["\n", "\n\n"]
        )
        ost_step_list = [io_output.strip() for io_output in io_output_list]

        #! generate potential answer to the user question
        potential_answers_list: List[List[str]] = []  # essentially direct answer list
        if self.enable_potential_score:
            for ost_step in ost_step_list:
                response_prefix = make_response_prefix(solution_trace, Node_Type.OST_STEP, new_ost_step=ost_step)

                potential_score_input = "Question: " + user_question + "\nAnswer: " + response_prefix

                potential_score_output = self.io.generate(
                    potential_score_input,
                    num_return=self.num_votes,
                    max_tokens=128,
                    stop_tokens=self.fewshot_cot_config["stop_tokens"],
                )
                potential_score_input2 = [
                    "Question: "
                    + user_question
                    + "\nAnswer: "
                    + response_prefix
                    + z
                    + "\nTherefore, the answer (arabic numerals) is"
                    for z in potential_score_output
                ]
                cleaned_io_output_list = self.io.generate(
                    potential_score_input2,
                    num_return=1,
                    max_tokens=128,
                    stop_tokens=self.fewshot_cot_config["stop_tokens"],
                )
                cleaned_io_output_list = [z[0] for z in cleaned_io_output_list]

                potential_answers_list.append(
                    [self.evaluator.extract_answer_from_model_completion(o) for o in cleaned_io_output_list]
                )
        else:
            potential_answers_list = [None] * len(ost_step_list)

        return ost_step_list, potential_answers_list


class Reasoning_MCTS_Node(MCTS_Node):
    def __init__(
        self,
        parent: "Reasoning_MCTS_Node",
        depth: int,
        node_type: Node_Type,
        verbose: bool = False,
        # --- For instantiating root node ---
        node_value: float = None,
        generator: Generator = None,
        disable_a5: bool = None,
        user_question: str = None,
        max_depth_allowed: int = None,
        disable_a1: bool = None,
        # -----------------------------------
        # --- For instantiating REPHRASED_USER_QUESTION node ---
        rephrased_user_question: str = None,
        # ------------------------------------------------------
        expected_answer: str = None,
        # --- For instantiating DIRECT_ANSWER node ---
        direct_answer: str = None,
        # --------------------------------------------
        # --- For instantiating SUBQUESTION node ---
        subquestion: str = None,
        subanswer: str = None,
        is_new_subquestion: bool = None,
        # ------------------------------------------
        # --- For instantiating RE_SUBANSWER node ---
        re_subanswer: str = None,
        # -------------------------------------------
        # --- For instantiating OST_STEP node ---
        ost_step: str = None,
        # ---------------------------------------
        # --- For node selection (not in sanity checks yet) ---
        enable_potential_score: bool = None,
        potential_answers: List[str] = None,
    ) -> None:
        """params:
        subquestion: the node is proposing a new subquestion
        subanswer: the answer corresponding to the new subquestion the node proposed
        re_subanswer: the node is proposing a new subanswer to the parent's subquestion
        """
        super().__init__()

        #! sanity checks
        try:
            assert depth is not None
            assert node_type is not None
            if node_value is not None:
                assert node_value > 0, breakpoint()

            if node_type is Node_Type.USER_QUESTION:
                assert depth == 0
                assert all(
                    attr is None
                    for attr in [
                        parent,
                        node_value,
                        rephrased_user_question,
                        direct_answer,
                        subquestion,
                        subanswer,
                        is_new_subquestion,
                        re_subanswer,
                        ost_step,
                    ]
                )
                assert all(
                    attr is not None
                    for attr in [generator, disable_a5, user_question, expected_answer, max_depth_allowed, disable_a1]
                )
            elif node_type is Node_Type.REPHRASED_USER_QUESTION:
                assert depth == 1
                assert all(
                    attr is None
                    for attr in [
                        node_value,
                        generator,
                        disable_a5,
                        user_question,
                        expected_answer,
                        direct_answer,
                        subquestion,
                        subanswer,
                        is_new_subquestion,
                        re_subanswer,
                        ost_step,
                        max_depth_allowed,
                        disable_a1,
                    ]
                )
                assert all(attr is not None for attr in [parent, rephrased_user_question])
            elif node_type is Node_Type.DIRECT_ANSWER:
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        generator,
                        disable_a5,
                        user_question,
                        expected_answer,
                        subquestion,
                        subanswer,
                        is_new_subquestion,
                        re_subanswer,
                        ost_step,
                        max_depth_allowed,
                        disable_a1,
                    ]
                )
                assert all(attr is not None for attr in [parent, node_value, direct_answer])
            elif node_type is Node_Type.SUBQUESTION:
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        generator,
                        disable_a5,
                        user_question,
                        expected_answer,
                        direct_answer,
                        re_subanswer,
                        ost_step,
                        max_depth_allowed,
                        disable_a1,
                    ]
                )
                assert all(
                    attr is not None for attr in [parent, node_value, subquestion, subanswer, is_new_subquestion]
                )
            elif node_type is Node_Type.RE_SUBANSWER:
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        generator,
                        disable_a5,
                        user_question,
                        expected_answer,
                        direct_answer,
                        subquestion,
                        subanswer,
                        is_new_subquestion,
                        ost_step,
                        max_depth_allowed,
                        disable_a1,
                    ]
                )
                assert all(attr is not None for attr in [parent, node_value, re_subanswer])
            elif node_type is Node_Type.OST_STEP:
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        node_value,
                        generator,
                        disable_a5,
                        user_question,
                        rephrased_user_question,
                        expected_answer,
                        direct_answer,
                        subquestion,
                        subanswer,
                        is_new_subquestion,
                        re_subanswer,
                        max_depth_allowed,
                        disable_a1,
                    ]
                )
                assert all(attr is not None for attr in [parent, ost_step])
        except AssertionError:
            print(f"Instantiating node with type {node_type} failed!")
            breakpoint()
            exit()

        #! attributes
        self.parent = parent  # if parent is None, then the node is the root
        self.children: List["Reasoning_MCTS_Node"] = []
        self.depth = depth
        self.node_type = node_type
        self.node_value = node_value
        self.direct_answer = direct_answer
        self.subquestion = subquestion
        self.subanswer = subanswer
        self.is_new_subquestion = is_new_subquestion
        self.re_subanswer = re_subanswer
        self.ost_step = ost_step

        if parent is None:  # root
            self.verbose = verbose
            self.user_question = user_question
            self.expected_answer = expected_answer
            self.generator = generator
            self.disable_a5 = disable_a5
            self.question_index = generator.question_index
            self.max_depth_allowed = max_depth_allowed
            self.disable_a1 = disable_a1
            self.enable_potential_score = enable_potential_score
        else:  # inherit from parent
            self.verbose = parent.verbose
            self.user_question = parent.user_question
            self.expected_answer = parent.expected_answer
            self.generator = parent.generator
            self.disable_a5 = parent.disable_a5
            self.question_index = parent.generator.question_index
            self.max_depth_allowed = parent.max_depth_allowed
            self.disable_a1 = parent.disable_a1
            self.enable_potential_score = parent.enable_potential_score

        #! keep track of paraphrasing
        if node_type is Node_Type.USER_QUESTION:
            self.paraphrased = False
        elif node_type is Node_Type.REPHRASED_USER_QUESTION:
            self.paraphrased = True
            self.user_question = rephrased_user_question
        else:
            assert parent is not None
            self.paraphrased = parent.paraphrased

        #! record number of subquestions till now
        if parent is None:  # root
            self.subquestion_counter = 0
        else:
            if node_type is Node_Type.SUBQUESTION and is_new_subquestion:
                self.subquestion_counter = parent.subquestion_counter + 1
            else:
                self.subquestion_counter = parent.subquestion_counter

        #! record number of one-step thought steps till now
        if parent is None:  # root
            self.ost_step_counter = 0
        else:
            if node_type is Node_Type.OST_STEP:
                self.ost_step_counter = parent.ost_step_counter + 1
            else:
                self.ost_step_counter = parent.ost_step_counter

        #! record solution trace from root to the current node. key: subquestion id
        if parent is None:  # root
            assert self.node_type is Node_Type.USER_QUESTION
            self.solution_trace: Dict[int, Dict[str, str]] = {0: {"user_question": user_question, "ost_step": {}}}
        else:
            assert self.node_type is not Node_Type.USER_QUESTION
            self.solution_trace = deepcopy(parent.solution_trace)

            if node_type is Node_Type.REPHRASED_USER_QUESTION:
                self.solution_trace[0]["user_question"] = rephrased_user_question
            elif node_type is Node_Type.DIRECT_ANSWER:
                assert self.subquestion_counter in self.solution_trace.keys()
                assert self.subquestion_counter == parent.subquestion_counter
                self.solution_trace[self.subquestion_counter]["direct_answer"] = {
                    "text": direct_answer,
                    "value": node_value,
                }
            elif node_type is Node_Type.SUBQUESTION:
                assert is_new_subquestion and self.subquestion_counter == parent.subquestion_counter + 1
                self.solution_trace[self.subquestion_counter] = {
                    "subquestion": subquestion,
                    "subanswer": {"text": subanswer, "value": node_value},
                    "ost_step": {},
                }
            elif node_type is Node_Type.RE_SUBANSWER:
                assert parent.subquestion is not None
                assert self.subquestion_counter == parent.subquestion_counter
                assert self.solution_trace[self.subquestion_counter]["subquestion"] == parent.subquestion
                self.solution_trace[self.subquestion_counter]["subanswer"] = {"text": re_subanswer, "value": node_value}
            elif node_type is Node_Type.OST_STEP:
                assert "ost_step" in self.solution_trace[self.subquestion_counter].keys()
                self.solution_trace[self.subquestion_counter]["ost_step"][self.ost_step_counter] = ost_step

        #! potential_score for intermediate nodes (only used for node selection)
        if self.enable_potential_score:
            self.potential_answers = potential_answers
            self.potential_score = 0
            if parent is None:  # root
                assert self.node_type is Node_Type.USER_QUESTION
                self.potential_answers_history = {}
            else:
                assert self.node_type is not Node_Type.USER_QUESTION
                self.potential_answers_history = deepcopy(parent.potential_answers_history)
                self.potential_answers_history[self.depth] = potential_answers

    def __str__(self) -> str:
        type2str = {
            Node_Type.USER_QUESTION: "U",
            Node_Type.REPHRASED_USER_QUESTION: "RU",
            Node_Type.DIRECT_ANSWER: "DA",
            Node_Type.SUBQUESTION: "SQ",
            Node_Type.RE_SUBANSWER: "RS",
            Node_Type.OST_STEP: "TS",
        }
        return f"{type2str[self.node_type]}-{self.id}"

    def _create_children(self):
        def do_action_generate_direct_answers():
            verbose_print(f"---- Generating direct answers for node {self.id}...", self.verbose)

            #! ACTION: generate direct answer for the user question (w/ or w/o hint)
            if (
                self.node_type is not Node_Type.USER_QUESTION
                and self.node_type is not Node_Type.REPHRASED_USER_QUESTION
            ):
                hint = make_hint(self.solution_trace, self.node_type)
            else:
                hint = None

            (direct_answer_list, value_list) = self.generator.generate_direct_answers(
                user_question=self.user_question, paraphrased=self.paraphrased, hint=hint
            )
            for direct_answer, value in zip(direct_answer_list, value_list):
                if np.isnan(value) or value <= 0:
                    breakpoint()
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.DIRECT_ANSWER,
                        node_value=value,
                        direct_answer=direct_answer,
                    )
                )

        def do_action_generate_subquestions():
            verbose_print(f"---- Generating subquestions for node {self.id}...", self.verbose)

            #! ACTION: generate new subquestions
            (subquestion_list, subanswer_list, value_list, potential_answers_list) = (
                self.generator.generate_subquestions(
                    user_question=self.user_question, solution_trace=self.solution_trace, paraphrased=self.paraphrased
                )
            )
            for subquestion, subanswer, value, potential_answers in zip(
                subquestion_list, subanswer_list, value_list, potential_answers_list
            ):
                if np.isnan(value) or value <= 0:
                    value = 0.01
                    # breakpoint()
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.SUBQUESTION,
                        node_value=value,
                        subquestion=subquestion,
                        subanswer=subanswer,
                        is_new_subquestion=True,
                        potential_answers=deepcopy(potential_answers),
                    )
                )

        def do_action_generate_re_subanswers():
            verbose_print(f"---- Generating re-subanswers for node {self.id}...", self.verbose)

            #! ACTION: re-generate subanswers for the previous subquestion
            (re_subanswer_list, value_list, potential_answers_list) = self.generator.generate_re_subanswers(
                user_question=self.user_question,
                solution_trace=self.solution_trace,
                paraphrased=self.paraphrased,
            )
            for re_subanswer, value, potential_answers in zip(re_subanswer_list, value_list, potential_answers_list):
                if np.isnan(value) or value <= 0:
                    breakpoint()
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.RE_SUBANSWER,
                        node_value=value,
                        re_subanswer=re_subanswer,
                        potential_answers=deepcopy(potential_answers),
                    )
                )

        def do_action_generate_rephrased_user_question():
            verbose_print(f"---- Generating rephrased user question for node {self.id}...", self.verbose)

            #! ACTION: generate paraphrased question for the root question
            rephrased_user_question_list, potential_answers_list = self.generator.generate_rephrased_user_question(
                user_question=self.user_question
            )
            for rephrased_user_question, potential_answers in zip(rephrased_user_question_list, potential_answers_list):
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.REPHRASED_USER_QUESTION,
                        rephrased_user_question=rephrased_user_question,
                        potential_answers=deepcopy(potential_answers),
                    )
                )

        def do_action_generate_ost_step(parent_is_subquestion=False):
            verbose_print(f"---- Generating one-step thought steps for node {self.id}...", self.verbose)

            #! ACTION: generate one-step thought step
            ost_step_list, potential_answers_list = self.generator.generate_ost_step(
                user_question=self.user_question,
                solution_trace=self.solution_trace,
                paraphrased=self.paraphrased,
                parent_is_subquestion=parent_is_subquestion,
            )
            for ost_step, potential_answers in zip(ost_step_list, potential_answers_list):
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.OST_STEP,
                        ost_step=ost_step,
                        potential_answers=deepcopy(potential_answers),
                    )
                )

        #! create children
        if self.node_type is Node_Type.USER_QUESTION:
            # A1: Propose an one-step thought.
            if not self.disable_a1:
                do_action_generate_ost_step()

            # A2: Propose the remaining thought steps
            do_action_generate_direct_answers()

            # A3: Propose next sub-question along with its answer.
            do_action_generate_subquestions()

            # A5: Rephrase the question/sub-question.
            if not self.disable_a5:
                do_action_generate_rephrased_user_question()

        elif self.node_type is Node_Type.REPHRASED_USER_QUESTION:
            # A1: Propose an one-step thought.
            if not self.disable_a1:
                do_action_generate_ost_step()

            # A2: Propose the remaining thought steps
            do_action_generate_direct_answers()

            # A3: Propose next sub-question along with its answer.
            do_action_generate_subquestions()
        elif self.node_type is Node_Type.DIRECT_ANSWER:
            raise ValueError("DIRECT_ANSWER node cannot create children!!")
        elif self.node_type is Node_Type.SUBQUESTION:
            # A1: Propose an one-step thought.
            if not self.disable_a1:
                do_action_generate_ost_step(parent_is_subquestion=True)

            # A2: Propose the remaining thought steps
            do_action_generate_direct_answers()

            # A3: Propose next sub-question along with its answer.
            do_action_generate_subquestions()

            # A4: Answer the sub-question again.
            do_action_generate_re_subanswers()
        elif self.node_type is Node_Type.RE_SUBANSWER:
            # A1: Propose an one-step thought.
            if not self.disable_a1:
                do_action_generate_ost_step(parent_is_subquestion=True)

            # A2: Propose the remaining thought steps
            do_action_generate_direct_answers()

            # A3: Propose next sub-question along with its answer.
            do_action_generate_subquestions()
        elif self.node_type is Node_Type.OST_STEP:
            # A1: Propose an one-step thought.
            if not self.disable_a1:
                do_action_generate_ost_step()

            # A2: Propose the remaining thought steps
            do_action_generate_direct_answers()

        assert self.children
        return self.children

    def is_valid_leaf_node(self):
        #! a valid solution can only be in SUBQUESTION type or DIRECT_ANSWER type
        return (
            self.node_type is Node_Type.SUBQUESTION and reach_terminal_subquestion(self.subquestion, self.user_question)
        ) or self.node_type is Node_Type.DIRECT_ANSWER

    def is_valid_solution_node(self):
        #! a valid solution can only be in SUBQUESTION type or DIRECT_ANSWER type or OST_STEP type
        return (
            (
                self.node_type is Node_Type.SUBQUESTION
                and reach_terminal_subquestion(self.subquestion, self.user_question)
            )
            or (self.node_type is Node_Type.OST_STEP and reach_terminal_ost_step(self.ost_step))
            or self.node_type is Node_Type.DIRECT_ANSWER
        )

    def set_potential_score(self, score: float):
        self.potential_score = score

    def find_children(self, rollout_id: int):
        self.children = self.children or self._create_children()
        for child in self.children:
            child.set_rollout_id(rollout_id)
        assert self.children
        return self.children

    def is_terminal(self):
        return self.depth >= self.max_depth_allowed or self.is_valid_leaf_node()

    def calculate_reward(self):
        if self.is_valid_leaf_node():
            assert self.node_value is not None, breakpoint()
            return self.node_value
        else:
            return 0

    def skip_backprop(self):
        return self.node_type is Node_Type.USER_QUESTION or self.node_type is Node_Type.REPHRASED_USER_QUESTION


def search_for_answers(args, user_question: str, question_id: int, gt_answer: str, generator: Generator):
    verbose_print(
        f"********************* Searching for answers to question {question_id} ********************* ", args.verbose
    )

    #! build an MCTS searcher
    mcts_searcher = MCTS_Searcher(
        exploration_weight=args.mcts_exploration_weight,
        weight_scheduler=args.mcts_weight_scheduler,
        num_rollouts=args.num_rollouts,
        discount=args.mcts_discount_factor,
        verbose=args.verbose,
    )

    #! build the MCTS tree
    root_node = Reasoning_MCTS_Node(
        parent=None,
        depth=0,
        node_type=Node_Type.USER_QUESTION,
        verbose=args.verbose,
        generator=generator,
        disable_a5=args.disable_a5,
        user_question=user_question,
        expected_answer=gt_answer,
        max_depth_allowed=args.max_depth_allowed,
        disable_a1=args.disable_a1,
        enable_potential_score=args.enable_potential_score,
    )

    model_solutions = []
    model_all_solutions = []
    model_rollout_nodes = []
    for i in (pbar := trange(args.num_rollouts, disable=True, position=0)):
        rollout_node = mcts_searcher.do_rollout(root_node, i)
        model_rollout_nodes.append(rollout_node)

        _, best_solution, _, chosen_node, all_solution_nodes, all_solutions = stochastic_find_best_solution(
            root_node, generator.evaluator, enable_potential_score=args.enable_potential_score
        )
        model_solutions.append(best_solution)
        model_all_solutions.append(all_solutions)

        if args.save_tree:
            with open(
                os.path.join(
                    args.answer_sheets_dir,
                    f"Question {question_id:04d} - Rollout {i}.tree",
                ),
                "w",
            ) as f:
                print_tree_from_root(
                    mcts_searcher=mcts_searcher,
                    rollout_id=i,
                    root_node=root_node,
                    chosen_node=chosen_node,
                    file=f,
                )

    #! record final traces
    js = [{"trace": node.solution_trace, "rollout_id": node.rollout_id} for node in all_solution_nodes]
    with open(os.path.join(args.answer_sheets_dir, f"Question {question_id:04d} - Final Solutions.json"), "w") as f:
        json.dump(js, f)

    js2 = [{"trace": node.solution_trace, "rollout_id": i} for i, node in enumerate(model_rollout_nodes)]
    with open(os.path.join(args.answer_sheets_dir, f"Question {question_id:04d} - Rollout Solutions.json"), "w") as f:
        json.dump(js2, f)

    if args.enable_potential_score:
        js = [node.potential_answers_history for node in all_solution_nodes]
        with open(os.path.join(args.answer_sheets_dir, f"Question {question_id:04d} - Potentials.json"), "w") as f:
            json.dump(js, f)

    return model_solutions, i, model_all_solutions
