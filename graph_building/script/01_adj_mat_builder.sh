# $1: probe list file path
# $2: rank list file path
# $3: rank size
# $4: image dir

script_path=$(readlink -f $0);
script_dir=$(dirname $script_path);
p1=''$script_dir'/../src/';

make_fold() {
	# $1: rank file path
	# $2: list size
	# $3: image dir
	cont=0;

	aux=$(basename $1)
	id=$(echo $aux | cut -c -32);
	mkdir $id
	mkdir ''$id'/imgs'

	for image in $(cut $1 -c -36); do
		if [ $cont -eq 0 ]; then
			cont=$(($cont + 1));

        elif [ $cont -lt $2 ]; then
        	cont=$(($cont + 1));
            path=$(find $3 -name '*'$image'*');
            cp $path ''$id'/imgs';
            echo $path;

            fn=$(basename $path);
            echo './imgs/'$fn'' >> ''$id'/img_list.txt';

        else break;
        fi;
	done;
}

date;
echo '01_adj_mat_builder.sh '$1' '$2' '$3' '$4'';
while read -r i && read -r j <&3; do
        b1=$(basename $i);
        b2=$(basename $j);

        id=$(echo $b1 | cut -c -32);
        echo 'Starting '$id' *******';
        date;

		make_fold $j $3 $4;
        cd $id;
        
        ip=$(find $4 -name '*'$b1'*');
        python ''$p1'/01_build_adj_mat.py' $ip img_list.txt '../'$b1'.npz';
        echo 'Finished '$id' *******';
        date;
        echo ' '

        cd ..;
        rm -r ''$id'/imgs';
done < $1 3<$2;
date;
