import os
import re
import json
import argparse
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
# from datasets import load_dataset


# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.enabled = False

from minigpt4.datasets.datasets.vqa_datasets import OKVQAEvalData, VizWizEvalData, IconQAEvalData, GQAEvalData, \
    VSREvalData, HMEvalData, OCRVQAEvalData, AOKVQAEvalDataset
from minigpt4.common.vqa_tools.VQA.PythonHelperTools.vqaTools.vqa import VQA
from minigpt4.common.vqa_tools.VQA.PythonEvaluationTools.vqaEvaluation.vqaEval import VQAEval
from minigpt4.datasets.datasets.ocrvqa_dataset import OCRVQADataset
from minigpt4.common.eval_utils import prepare_texts, init_model, eval_parser
from minigpt4.conversation.conversation import CONV_VISION_minigptv2,CONV_VISION_Vicuna0
from minigpt4.conversation.conversation import CONV_VISION_minigptv2,CONV_VISION_Vicuna0
from minigpt4.common.config import Config


def list_of_str(arg):
    return list(map(str, arg.split(',')))


parser = eval_parser()
parser.add_argument("--dataset", type=list_of_str, default='refcoco', help="dataset to evaluate")
args = parser.parse_args()
cfg = Config(args)

model, vis_processor = init_model(args)
# conv_temp = CONV_VISION_minigptv2.copy()
conv_temp = CONV_VISION_Vicuna0.copy()
model.eval()
save_path = cfg.run_cfg.save_path

if 'okvqa' in args.dataset:

    eval_file_path = cfg.evaluation_datasets_cfg["okvqa"]["eval_file_path"]
    img_path = cfg.evaluation_datasets_cfg["okvqa"]["img_path"]
    batch_size = cfg.evaluation_datasets_cfg["okvqa"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["okvqa"]["max_new_tokens"]

    evaluation_annntation_path = os.path.join(eval_file_path, "okvqa_test_split.json")
    with open(evaluation_annntation_path) as f:
        ok_vqa_test_split = json.load(f)

    data = OKVQAEvalData(ok_vqa_test_split, vis_processor, img_path)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    minigpt4_predict = []

    for images, questions, question_ids, img_ids in eval_dataloader:
        texts = prepare_texts(questions, conv_temp)  # warp the texts with conversation template
        answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)

        for answer, question_id, question, img_id in zip(answers, question_ids, questions, img_ids):
            result = dict()
            answer = answer.lower().replace('<unk>', '').strip()
            answer = answer.split("###human:")[0]# delete prompt
            result['answer'] = answer
            result['question_id'] = int(question_id)
            minigpt4_predict.append(result)

    file_save_path = os.path.join(save_path, "okvqa.json")
    with open(file_save_path, 'w') as f:
        json.dump(minigpt4_predict, f)

    annFile = os.path.join(eval_file_path, "mscoco_val2014_annotations_clean.json")
    quesFile = os.path.join(eval_file_path, "OpenEnded_mscoco_val2014_questions_clean.json")

    vqa = VQA(annFile, quesFile)
    vqaRes = vqa.loadRes(file_save_path, quesFile)

    vqaEval = VQAEval(vqa, vqaRes, n=2)
    vqaEval.evaluate()
    print("Overall OKVQA Accuracy is: %.02f\n" % (vqaEval.accuracy['overall']), flush=True)

if 'aok_vqa' in args.dataset:
    eval_file_path = cfg.evaluation_datasets_cfg["aok_vqa"]["eval_file_path"]
    img_path = cfg.evaluation_datasets_cfg["aok_vqa"]["img_path"]
    batch_size = cfg.evaluation_datasets_cfg["aok_vqa"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["aok_vqa"]["max_new_tokens"]

    evaluation_annntation_path = os.path.join(eval_file_path, "aokvqa_v1p0_val.json") #"aokvqa_v1p0_test.json"
    with open(evaluation_annntation_path) as f:
        aok_vqa_test_split = json.load(f)
    data = AOKVQAEvalDataset(aok_vqa_test_split, vis_processor, img_path)#vis_processor, text_processor, vis_root, ann_paths
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    minigpt4_predict = []
    count = 0
    total = 0

    # data_iter = iter(eval_dataloader)
    # batch_data = next(data_iter)
    # print("第一次测试")
    # print(type(batch_data))
    # print(batch_data)

    # image = self.vis_processor(image)
    # question = self.text_processor(ann["question"])
    # answer_key = "direct_answers"
    # answer_weight = {}
    # for answer in ann[answer_key]:
    #     if answer in answer_weight.keys():
    #         answer_weight[answer] += 1 / len(ann[answer_key])
    #     else:
    #         answer_weight[answer] = 1 / len(ann[answer_key])
    # answers = list(answer_weight.keys())
    # weights = list(answer_weight.values())
    # answer = random.choices(answers, weights=weights, k=1)[0]

    # for images, questions, question_ids,choices, correct_choice_idx, direct_answers in eval_dataloader:
    for images, questions, question_ids, direct_answers in eval_dataloader:
        texts = prepare_texts(questions, conv_temp)  # warp the texts with conversation template
        answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)
        for answer,question_id, question, direct_answer in zip(answers, question_ids, questions, direct_answers):
            result = dict()
            answer = answer.lower().replace('<unk>', '').strip()
            answer = answer.split("###human:")[0]# delete prompt
            result['answer'] = answer
            result['direct_answers'] = direct_answer
            result['question_id'] = question_id
            minigpt4_predict.append(result)
            if answer in direct_answer:
                count+=1
            total+=1

    file_save_path = os.path.join(save_path, "aokvqa.json")
    with open(file_save_path, 'w') as f:
        json.dump(minigpt4_predict, f)

    print('aok val:', count / total * 100, flush=True)

    # pred_qa_pairs = []
    # question_id = samples["question_id"]
    # gt_answers = samples["direct_answers"]
    # 
    # for pred_answer, ques_id, gt_answer in zip(answers, question_id, gt_answers):
    #     pred_qa_pairs.append(
    #         {"question_id": ques_id, "pred_ans": pred_answer, "gt_ans": gt_answer}
    #     )
    # return pred_qa_pairs


if 'ocrvqa' in args.dataset:
    eval_file_path = cfg.evaluation_datasets_cfg["ocrvqa"]["eval_file_path"]
    root_path = cfg.evaluation_datasets_cfg["ocrvqa"]["img_path"]
    batch_size = cfg.evaluation_datasets_cfg["ocrvqa"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["ocrvqa"]["max_new_tokens"]

    with open(eval_file_path) as f:
        ocr_vqa_test_split = json.load(f)
    data = OCRVQADataset.create_data(root_path,eval_file_path)
    # data = OCRVQAEvalData(ocr_vqa_test_split, vis_processor, root_path)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    minigpt4_predict = []
    count = 0
    total = 0

    # data_iter = iter(eval_dataloader)
    # batch_data = next(data_iter)
    # print("第一次测试")
    # print(type(batch_data))
    # print(batch_data)

    for batch in eval_dataloader:
        questions, gt_answers, image_paths, imags_ids, titles, genres = batch["question"], batch["answer"],batch["image_path"],batch["image_id"],batch["title"],batch["genre"]
        # image process
        tensors = []
        for image_path in image_paths:
            image = Image.open(os.path.join(root_path, image_path)).convert("RGB")
            image = vis_processor(image)
            tensors.append(image.unsqueeze(0))
        images = torch.cat(tensors, dim=0)
        # pred answers
        # print(images.size())# torch.tensor[10, 3, 224, 224]
        # print(len(questions))# tuple, len=10
        texts = prepare_texts(questions, conv_temp)  # warp the texts with conversation template
        answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)
        for question, gt_answer, answer in zip(questions, gt_answers, answers):
            result = dict()
            answer = answer.lower().replace('<unk>', '').strip()
            answer = answer.split("###human:")[0]# delete prompt
            result['question'] = question
            result['answer'] = answer
            result['gt_answer'] = gt_answer
            minigpt4_predict.append(result)
            if answer.lower()==gt_answer.lower():
                count+=1
            total+=1
            print("count / total",count, total)


    file_save_path = os.path.join(save_path, "ocrvqa.json")
    with open(file_save_path, 'w') as f:
        json.dump(minigpt4_predict, f)
    print('OCR Acc: ', count / total * 100, flush=True)

if 'vizwiz' in args.dataset:

    eval_file_path = cfg.evaluation_datasets_cfg["vizwiz"]["eval_file_path"]
    img_path = cfg.evaluation_datasets_cfg["vizwiz"]["img_path"]
    batch_size = cfg.evaluation_datasets_cfg["vizwiz"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["vizwiz"]["max_new_tokens"]

    vizwiz = json.load(open(eval_file_path, 'r'))

    data = VizWizEvalData(vizwiz, vis_processor, img_path)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    minigpt4_predict = []
    total_acc = []
    for images, texts, gt_answers in tqdm(eval_dataloader):
        texts = prepare_texts(texts, conv_temp)  # warp the texts with conversation template
        with torch.no_grad():
            answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False,
                                     repetition_penalty=1.0)

        for answer, gt_answer in zip(answers, gt_answers):
            answer=answer.lower().replace('<unk>', '').strip()
            answer=answer.split("###human:")[0]  # delete prompt
            result = dict()
            result['answer'] = answer
            result['gt_answer'] = gt_answer
            minigpt4_predict.append(result)
            count = 0
            gt_answer = gt_answer.split('_')
            for gt in gt_answer:
                if gt.lower() in answer:
                    count += 1
            acc = min(count / 3.0, 1.0)
            total_acc.append(acc)

    file_save_path = os.path.join(save_path, "vizwiz.json")
    with open(file_save_path, 'w') as f:
        json.dump(minigpt4_predict, f)
    print('vizwiz Acc: ', np.average(total_acc) * 100.0, flush=True)


if 'iconvqa' in args.dataset:

    eval_file_path = cfg.evaluation_datasets_cfg["iconvqa"]["eval_file_path"]
    img_path = cfg.evaluation_datasets_cfg["iconvqa"]["img_path"]
    batch_size = cfg.evaluation_datasets_cfg["iconvqa"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["iconvqa"]["max_new_tokens"]

    iconqa_text_val = json.load(open(eval_file_path, "r"))

    data = IconQAEvalData(iconqa_text_val, vis_processor, img_path)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)

    count = 0
    for images, texts, candidates, answers in tqdm(eval_dataloader):
        candidates = [candidate.split('_') for candidate in candidates]
        num_cand = [len(candidate) for candidate in candidates]
        for candidate in candidates:
            candidate.extend(['none'] * (max(num_cand) - len(candidate)))
        candidates = [list(x) for x in zip(*candidates)]
        instructions = ["<s>[INST] <Img><ImageHere></Img> {} [/INST]".format(text) for text in texts]
        answer_ranks = model.multi_select(images, instructions, candidates, num_cand=num_cand)
        for idx, answer in enumerate(answers):
            if answer_ranks[idx][0] == answer:
                count += 1

    print('iconqa Acc: ', count / len(iconqa_text_val) * 100.0, flush=True)

if 'gqa' in args.dataset:

    eval_file_path = cfg.evaluation_datasets_cfg["gqa"]["eval_file_path"]
    img_path = cfg.evaluation_datasets_cfg["gqa"]["img_path"]
    batch_size = cfg.evaluation_datasets_cfg["gqa"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["gqa"]["max_new_tokens"]

    # evaluation_annntation_path = os.path.join(eval_file_path, "test_balanced_questions.json")
    evaluation_annntation_path = os.path.join(eval_file_path, "testdev_balanced_questions.json")
    gqa = json.load(open(evaluation_annntation_path))
    data = GQAEvalData(gqa, vis_processor, img_path)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    count = 0
    total = 0
    minigpt4_predict = []
    for images, texts, labels in tqdm(eval_dataloader):
        texts = prepare_texts(texts, conv_temp)  # warp the texts with conversation template
        answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)

        for answer, label in zip(answers, labels):
            result = dict()
            answer = answer.lower().replace('<unk>', '').strip()
            answer = answer.split("###human")[0]
            result['pred'] = answer
            result['gt'] = label
            minigpt4_predict.append(result)
            if answer.lower() == label:
                count += 1
            total += 1
    print('gqa val:', count / total * 100, flush=True)

    file_save_path = os.path.join(save_path, "gqa.json")
    with open(file_save_path, 'w') as f:
        json.dump(minigpt4_predict, f)

# if 'vsr' in args.dataset:
#
#     img_path = cfg.evaluation_datasets_cfg["vsr"]["img_path"]
#     batch_size = cfg.evaluation_datasets_cfg["vsr"]["batch_size"]
#     max_new_tokens = cfg.evaluation_datasets_cfg["vsr"]["max_new_tokens"]
#
#     annotation = load_dataset("cambridgeltl/vsr_zeroshot", split='test')
#     data = VSREvalData(annotation, vis_processor, img_path)
#     eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
#     count=0
#     total=0
#
#     minigpt4_predict = []
#
#     for images, texts, labels in tqdm(eval_dataloader):
#         texts = prepare_texts(texts, conv_temp)  # warp the texts with conversation template
#         answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)
#
#         for answer, label in zip(answers, labels):
#             result = dict()
#             result['pred'] = answer.replace('<unk>','').strip()
#             result['gt'] = label
#             minigpt4_predict.append(result)
#             if answer.lower() ==  label.lower():
#                 count+=1
#             total+=1
#     print('vsr test:', count / total * 100, flush=True)
#     file_save_path = os.path.join(save_path,"vsr.json")
#     with open(file_save_path,'w') as f:
#         json.dump(minigpt4_predict, f)

if 'hm' in args.dataset:

    eval_file_path = cfg.evaluation_datasets_cfg["hm"]["eval_file_path"]
    img_path = cfg.evaluation_datasets_cfg["hm"]["img_path"]
    batch_size = cfg.evaluation_datasets_cfg["hm"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["hm"]["max_new_tokens"]

    annotation = []
    with open(eval_file_path, 'r') as jsonl_file:
        for line in jsonl_file:
            json_obj = json.loads(line)
            annotation.append(json_obj)

    data = HMEvalData(annotation, vis_processor, img_path)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    count = 0
    total = 0

    minigpt4_predict = []

    for images, texts, labels in tqdm(eval_dataloader):
        texts = prepare_texts(texts, conv_temp)  # warp the texts with conversation template

        answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)

        for answer, label in zip(answers, labels):
            result = dict()
            if answer.lower().strip() == "yes":
                answer = 1
            elif answer.lower().strip() == "no":
                answer = 0
            else:
                print("non-matching answer", answer)

            result['pred'] = answer
            result['gt'] = int(label)
            minigpt4_predict.append(result)
            if answer == label:
                count += 1
            total += 1

    print('hm val:', count / total * 100, flush=True)
    file_save_path = os.path.join(save_path, "hm.json")
    with open(file_save_path, 'w') as f:
        json.dump(minigpt4_predict, f)
