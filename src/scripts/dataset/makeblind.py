import json

examples = []
with open("data/fever-data/shared_task_dev.jsonl") as input_data:
    for line in input_data:
        examples.append(json.loads(line))



with open("data/fever-data/example_blind_public.jsonl","w+") as public, open("data/fever-data/example_blind_private.jsonl","w+") as private:
    for line in examples:
        public.write(json.dumps({"id": line['id'], "claim": line["claim"]})+"\n")
        private.write(json.dumps({"id": line['id'], "evidence": line["evidence"],"label":line["label"]})+"\n")
