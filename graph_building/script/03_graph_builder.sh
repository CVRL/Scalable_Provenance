# $1: rank list file path
# $2: npz list file path
# $3: build type (0: kruskal, 1: cluster)
# $4: output dir path

script_path=$(readlink -f $0);
script_dir=$(dirname $script_path);
p1=''$script_dir'/../src/';

date;
echo '03_graph_builder.sh '$1' '$2' '$3' '$4'';
while read -r i && read -r j <&3; do
        b=$(basename $j);
        id=$(echo $b | cut -c -32);
        python ''$p1'/03_build_graph.py' $3 $i $j 0 ''$4'/'$id'.json';
done < $1 3<$2;
date;
