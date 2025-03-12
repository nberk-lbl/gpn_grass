# run from terminal

from huggingface_hub import HfApi
from huggingface_hub import login
api = HfApi()

login() # requires huggingface token

private = False
repo_id = "nberkowitz/gpn_grass"  # replace with your username, dataset name
folder_path = "/pscratch/sd/n/nberk/results/dataset"
api.create_repo(repo_id=repo_id, repo_type="dataset", private=private)
api.upload_folder(repo_id=repo_id, folder_path=folder_path, repo_type="dataset")