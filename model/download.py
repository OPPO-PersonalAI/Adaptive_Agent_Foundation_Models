from huggingface_hub import snapshot_download

# you can change to our other models in "https://huggingface.co/PersonalAILab/models"
snapshot_download(
    repo_id="PersonalAILab/A2FM-32B-rl",
    local_dir="./A2FM-32B-rl",
    local_dir_use_symlinks=False,
    resume_download=True
)