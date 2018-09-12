# $1: npz list file path
# $2: graph ids for joining (format: idx0,idx1,idx2,...,idxn)
# $3: output dir

script_path=$(readlink -f $0);
script_dir=$(dirname $script_path);
p1=''$script_dir'/../src/';

date;
echo '02_adj_mat_joiner.sh '$1' '$2' '$3'';

mkdir -p $3;

for i in $(cat $1); do
	bi=$(basename $i);
	output=''$3'/'$bi'.join.npz';

	python ''$p1'/02_join_adj_mat.py' $i $2 $output;
	echo 'Finished '$i' *******';
done;
date;
