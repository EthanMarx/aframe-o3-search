#!/bin/bash

# query gate files
#python zenodo_get uv run zenodo_get -g '*GATES*' 5636795 --output_dir ./data/gates/
#
## apply vetos to raw trigger files
#python ./aframe_o3_search/scripts/apply_vetos.py \
#    --data_dir ./data/aframe/pre_veto --out_dir ./data/aframe/post_veto \
#    --veto_definer_file ./aframe/projects/plots/plots/vetos/H1L1-HOFT_C01_O3_CBC.xml \
#    --gates '{"H1" : "./data/gates/H1-O3_GATES_1238166018-31197600.txt", "L1" : "./data/gates/L1-O3_GATES_1238166018-31197600.txt"}' 

# create cumulative histogram and pastro plot
#python ./aframe_o3_search/scripts/figure_4.py \
#    --background_path ./data/aframe/post_veto/background.hdf5 \
#    --foreground_path ./data/aframe/post_veto/foreground.hdf5 \
#    --rejected_path ./data/aframe/pre_veto/rejected-parameters.hdf5 \
#    --outdir ./paper/

#python ./aframe_o3_search/scripts/tables_1_and_2.py \
#    --segments_dir ./data/aframe/pre_veto \
#    --background_path ./data/aframe/post_veto/background.hdf5 \
#    --zero_lag_path ./data/aframe/post_veto/0lag.hdf5 \
#    --foreground_path ./data/aframe/post_veto/foreground.hdf5 \
#    --rejected_path ./data/aframe/pre_veto/rejected-parameters.hdf5 \
#    --pipeline_data_dir ./data/ \
#    --data_dir ./data/aframe/post_veto \
#    --out_dir ./paper/ --recovery_dt 1.0

python ./aframe_o3_search/scripts/figure_3.py --catalog_path ./paper/gwtc3.hdf5 --outdir ./paper/