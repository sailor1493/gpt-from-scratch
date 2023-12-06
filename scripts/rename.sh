#!/bin/bash

src_dir=/data/s1/chanwoo/nlp_project/logs/mixup_/data/s1/chanwoo/nlp_project/experiment_corpus
dest_dir=/data/s1/chanwoo/nlp_project/logs/

dirs=$(ls $src_dir)
for dir in $dirs
do
    echo $dir
    src=$src_dir/$dir
    # replace ".json" to ""
    new_dirname=${dir//.json/}
    dest=$dest_dir$new_dirname
    echo $src
    echo $dest
    mv $src $dest
done