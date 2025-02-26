# %% [markdown]
# Create a config.json in the following format
# ```
# {
#     "subscription_id": '<subscription-id>',
#     "resource_group": '<resource-group>',
#     "workspace_name": '<workspace-name>'
# }
# ```
# 

# %%
from azureml.core import Workspace
ws = Workspace.from_config()

# %%
from azureml.core import Environment
fastai_env = Environment("fastai2")

# %%
fastai_env.docker.base_image = "fastdotai/fastai2:latest"
fastai_env.python.user_managed_dependencies = True

# %% [markdown]
# ### Use a custom Dockerfile (optional)

# %%
# Specify Docker steps as a string
dockerfile = r"""
FROM mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04:20210615.v1
RUN echo "Hello from custom container!"
"""

# Set the base image to None, because the image is defined by Dockerfile
fastai_env.docker.base_image = None
fastai_env.docker.base_dockerfile = dockerfile

# Alternatively, load the string from a file
fastai_env.docker.base_image = None
fastai_env.docker.base_dockerfile = "./Dockerfile"

# %% [markdown]
# ### Attach a compute target

# %%
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

# Name of target cluster
cluster_name = "graph-gpu"

compute_target = ComputeTarget(workspace=ws, name=cluster_name)
print('Found existing compute target.')

# Use get_status() to get a detailed status for the current AmlCompute
print(compute_target.get_status().serialize())

# %%
from azureml.core import ScriptRunConfig

src = ScriptRunConfig(source_directory='fastai-example',
                      script='train.py',
                      compute_target=compute_target,
                      environment=fastai_env)

# %%



