import sys
sys.path.append("../datasets")
from data_utils import plot_xy_from_csv

'''
this is used to track the progress of training. run this from the current directory
'''

# model = "vqvae"
model = "maskgit"

plot_path = "../figures"
file_path = f"../figures/{model}_trans/train_val_accuracy.csv"
plot_xy_from_csv(file_path, title="trainval_accuracy", x_label="epochs", y_label="accuracy", name=f"{model}_trainval_accuracy", plot_path=plot_path, vis=False)

file_path = f"../figures/{model}_trans/train_val_loss.csv"
plot_xy_from_csv(file_path, title="trainval_loss", x_label="epochs", y_label="loss", name=f"{model}_trainval_loss", plot_path=plot_path, vis=False)

# file_path = f"../figures/{model}_trans/train_val_precision.csv"
# plot_xy_from_csv(file_path, title="trainval_precision", x_label="epochs", y_label="precision", name=f"{model}_trainval_precision", plot_path=plot_path, vis=False)

# file_path = f"../figures/{model}_trans/train_val_recall.csv"
# plot_xy_from_csv(file_path, title="trainval_recall", x_label="epochs", y_label="recall", name=f"{model}_trainval_recall", plot_path=plot_path, vis=False)