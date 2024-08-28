t_len=10623
step=20
i_size=532
for j in $(seq 0 $((i_size - 1))); do
    python longformer.py -i $j
done