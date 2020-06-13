from Config import *
from RMeN import *
import json
import os


from argparse import ArgumentParser
parser = ArgumentParser("RMeN")
parser.add_argument("--dataset", default="WN18RR", help="Name of the dataset.")
parser.add_argument("--learning_rate", default=0.05, type=float, help="Learning rate")
parser.add_argument("--nbatches", default=100, type=int, help="Number of batches")
parser.add_argument("--num_epochs", default=400, type=int, help="Number of training epochs")
parser.add_argument("--model_name", default='RMeN', help="")
parser.add_argument('--neg_num', default=1, type=int, help='')
parser.add_argument('--hidden_size', type=int, default=32, help='')
parser.add_argument('--num_of_filters', type=int, default=100, help='')
parser.add_argument('--dropout', type=float, default=0.5, help='')
parser.add_argument('--save_steps', type=int, default=1000, help='')
parser.add_argument('--valid_steps', type=int, default=100, help='')
parser.add_argument("--lmbda", default=0.1, type=float, help="")
parser.add_argument("--lmbda2", default=0.01, type=float, help="")
parser.add_argument("--mode", choices=["train", "predict"], default="train", type=str)
parser.add_argument("--checkpoint_path", default=None, type=str)
parser.add_argument("--test_file", default="", type=str)
parser.add_argument("--optim", default='adagrad', help="")
parser.add_argument("--num_heads", default=2, type=int, help="Number of attention heads. 1 2 4")
parser.add_argument("--memory_slots", default=1, type=int, help="Number of memory slots. 1 2 4")
parser.add_argument("--head_size", default=64, type=int, help="")
parser.add_argument("--gate_style", default='memory', help="unit,memory")
parser.add_argument("--attention_mlp_layers", default=2, type=int, help="2 3 4")
parser.add_argument("--use_pos", default=1, type=int, help="1 when using positional embeddings. Otherwise.")

args = parser.parse_args()

if args.model_name is None or len(args.model_name.strip()) == 0:
    args.model_name = "{}_lda-{}_nneg-{}_hs-{}_lr-{}_nepochs-{}".format(args.dataset,
                                                                        args.lmbda,
                                                                        args.neg_num,
                                                                        args.hidden_size,
                                                                        args.learning_rate,
                                                                        args.num_epochs)
print(args)

out_dir = os.path.abspath(os.path.join("../runs_RMeN/"))
print("Writing to {}\n".format(out_dir))
# Checkpoint directory
checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
result_dir = os.path.abspath(os.path.join(checkpoint_dir, args.model_name))
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

con = Config()
in_path = "./benchmarks/" + args.dataset + "/"
con.set_in_path(in_path)
test_file_path = ""
if args.test_file != "":
    test_file_path = in_path + args.test_file
con.set_test_file_path(test_file_path)
con.set_work_threads(8)
con.set_train_times(args.num_epochs)
con.set_nbatches(args.nbatches)
con.set_alpha(args.learning_rate)
con.set_bern(1)
con.set_dimension(args.hidden_size)
con.set_lmbda(args.lmbda)
con.set_lmbda_two(0.01)
con.set_margin(1.0)
con.set_ent_neg_rate(args.neg_num)
con.set_opt_method(args.optim)
con.set_save_steps(args.save_steps)
con.set_valid_steps(args.valid_steps)
con.set_early_stopping_patience(10)
con.set_checkpoint_dir(checkpoint_dir)
con.set_result_dir(result_dir)
#set True for the knowledge graph completion task
con.set_test_link(True)
con.init()

if args.mode == "train":
    # con.set_init_embeddings(entity_embs, rel_embs)
    con.set_config_CNN(num_of_filters=args.num_of_filters, drop_prob=args.dropout,
                mem_slots=args.memory_slots, head_size=args.head_size, num_heads=args.num_heads,
                attention_mlp_layers=args.attention_mlp_layers, use_pos=args.use_pos, gate_style='memory')

    con.set_train_model(RMeN)
    con.training_model()

else:
    con.set_config_CNN(num_of_filters=args.num_of_filters, drop_prob=args.dropout,
                       mem_slots=args.memory_slots, head_size=args.head_size, num_heads=args.num_heads,
                       attention_mlp_layers=args.attention_mlp_layers, use_pos=args.use_pos, gate_style='memory')

    con.set_test_model(RMeN, args.checkpoint_path)
    con.test()

