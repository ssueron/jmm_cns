from s4dd import S4forDenovoDesign
from s4dd.torch_callbacks import EarlyStopping, ModelCheckpoint

ckpt_dir = "s4-for-de-novo-drug-design/chembl_pretrained"  # folder holding model.pt and init_arguments.json
s4 = S4forDenovoDesign.from_file(ckpt_dir)

s4.n_max_epochs = 200          # set your budget
s4.batch_size = 256           # drop if you see OOM
s4.learning_rate = 1e-3       # or lower for finetuning
s4.device = "cuda"            # use "cpu" if no GPU

callbacks = [
    EarlyStopping(patience=5, delta=1e-4, criterion="val_loss", mode="min"),
    ModelCheckpoint(save_fn=s4.save, save_per_epoch=5, basedir="./models/")
]

history = s4.train(
    training_molecules_path="data/train.zip",   # or "data/train.txt"
    val_molecules_path="data/valid.zip",
    callbacks=callbacks,
)
s4.save("./models/finetuned/")
