GPU=$gpu PYTHONPATH=src:lib/DrQA python src/scripts/rte/mlp/dr_fever_riedel_3way.py 100
GPU=$gpu PYTHONPATH=src:lib/DrQA python src/scripts/rte/mlp/dr_fever_riedel_3way.py 10
GPU=$gpu PYTHONPATH=src:lib/DrQA python src/scripts/rte/mlp/dr_fever_riedel_3way.py 5
GPU=$gpu PYTHONPATH=src:lib/DrQA python src/scripts/rte/mlp/dr_fever_riedel_3way.py 1

GPU=$gpu PYTHONPATH=src:lib/DrQA python src/scripts/rte/mlp/dr_fever_riedel_3way_drqa.py 100 1
GPU=$gpu PYTHONPATH=src:lib/DrQA python src/scripts/rte/mlp/dr_fever_riedel_3way_drqa.py 10 1
GPU=$gpu PYTHONPATH=src:lib/DrQA python src/scripts/rte/mlp/dr_fever_riedel_3way_drqa.py 5 1
GPU=$gpu PYTHONPATH=src:lib/DrQA python src/scripts/rte/mlp/dr_fever_riedel_3way_drqa.py 1 1