import csv
import json
from sklearn.model_selection import train_test_split

# 读取CSV文件并转换为JSON格式
def csv_to_json(csv_filename, json_filename):
    with open(csv_filename, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        data = [row for row in csv_reader]  # 读取所有行并存储为字典列表

    # 将数据写入到JSON文件
    with open(json_filename, mode='w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)




if __name__ == "__main__":
    # region CulturalBench
    dir = 'dataset/'
    dir_json = 'dataset_JSON/'

    CulturalBench_csv = dir + 'CulturalBench-Hard.csv'  # 输入的CSV文件路径
    CulturalBench_json = dir_json + 'CulturalBench-Hard.json'  # 输出的JSON文件路径
    csv_to_json(CulturalBench_csv, CulturalBench_json)

    CulturalBench_v2_json = dir_json + 'CulturalBench-Hard_v2.json'
    with open(CulturalBench_json, 'r', encoding="utf-8") as f:
        data = json.load(f)
    for entry in data:
        entry['prompt_question_country'] = entry['prompt_question'] + " The country corresponding to this question is " + entry['country'] +"."  # 拼接
    with open(CulturalBench_v2_json, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    with open(CulturalBench_v2_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    prompts = [item["prompt_question_country"] for item in data]
    queries = [item["prompt_option"] for item in data]
    responses = [item["answer"] for item in data]
    train_prompts, test_prompts, train_queries, test_queries, train_responses, test_responses = train_test_split(
        prompts, queries, responses, test_size=0.1, random_state=42)
    train_data = [{"prompt": p, "query": q, "response": r} for p, q, r in
                  zip(train_prompts, train_queries, train_responses)]
    test_data = [{"prompt": p, "query": q, "response": r} for p, q, r in
                 zip(test_prompts, test_queries, test_responses)]
    CulturalBench_train_json = dir_json + 'CulturalBench-Hard_train.json'
    CulturalBench_test_json = dir_json + 'CulturalBench-Hard_test.json'
    with open(CulturalBench_train_json, "w", encoding="utf-8") as f:
        json.dump(train_data, f)
    with open(CulturalBench_test_json, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    # endregion

    # region global_opinions
    global_opinions_csv = dir + 'global_opinions.csv'  # 输入的 CSV 文件路径
    global_opinions_json = dir_json + 'global_opinions.json'  # 输出的 JSON 文件路径
    csv_to_json(global_opinions_csv, global_opinions_json)

    global_opinions_v2_json = dir_json + 'global_opinions_v2.json'
    with open(global_opinions_json, 'r', encoding="utf-8") as f:
        data = json.load(f)
    for entry in data:
        entry['question_source'] = entry['question'] + " The source of this problem is " + entry['source'] + "."  # 拼接
    with open(global_opinions_v2_json, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    with open(global_opinions_v2_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    prompts = [item["question_source"] for item in data]
    queries = [item["options"] for item in data]
    responses = [item["selections"] for item in data]
    train_prompts, test_prompts, train_queries, test_queries, train_responses, test_responses = train_test_split(
        prompts, queries, responses, test_size=0.1, random_state=42)
    train_data = [{"prompt": p, "query": q, "response": r} for p, q, r in
                  zip(train_prompts, train_queries, train_responses)]
    test_data = [{"prompt": p, "query": q, "response": r} for p, q, r in
                 zip(test_prompts, test_queries, test_responses)]
    with open("dataset_JSON/global_opinions_train.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f)
    with open("dataset_JSON/global_opinions_test.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    # endregion

    # region culturebank
    culturebank_csv = dir + 'culturebank_tiktok.csv'
    culturebank_json = dir_json + 'culturebank_tiktok.json'
    csv_to_json(culturebank_csv, culturebank_json)

    culturebank_v2_json = dir_json + 'culturebank_tiktok_v2.json'
    with open(culturebank_json, 'r', encoding="utf-8") as f:
        data = json.load(f)
    for entry in data:
        entry['background'] = ("Task Background: " +
                               "Cultural Group Targeted by the Behavior:" + entry['cultural group'] +
                               ". Context in Which the Behavior Occurs: " + entry['context'] +
                               ". Goal of the Behavior: " + entry['goal'] +
                               ". Relation of the Behavior to Other Factors: " + entry['relation'] +
                               ". Actor of the Behavior: " + entry['actor'] +
                               ". Actor's Behavior: " + entry['actor_behavior'] +
                               ". Recipient of the Behavior: " + entry['recipient'] +
                               ". Recipient's Behavior: " + entry['recipient_behavior'] +
                               ". Other Supplementary Descriptions:" + entry['other_descriptions'] +
                               ". Topic of the Behavior: " + entry['topic'] +
                               ". Agreement of the Behavior:" + entry['agreement'] +
                               ". Quantile Range of Support for a Behavior: " + entry['num_support_bin'] +
                               ". Time Range of the Behavior in Different Years: " + entry['time_range'] +
                               ". Problem Description: "+
                               "  Whole Description of the Problem Behavior: " + entry['eval_whole_desc'] +
                               ". Scenario of the Problem Behavior: " + entry['eval_scenario'] +
                               ". Relevant persona: " + entry['eval_persona'])
    with open(culturebank_v2_json, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    # endregion





