# Fixed-clocks test. Saves the clocks, fixes them at the CURRENT nvpmodel (15 W), repeats the matrix,
# restores on any exit and compares `jetson_clocks --show` before and after. Never touches nvpmodel.
cd $REPO_DIR
C=out/clocks; mkdir -p $C
idle() {  # idle TAG SECONDS: board power with nothing running
  sudo -n tegrastats --interval 1000 --logfile $C/idle_$1.log > /dev/null 2>&1 &
  sleep $2; sudo -n tegrastats --stop > /dev/null 2>&1; sleep 1
}
restore() {
  if [ -f $C/clocks_orig.conf ]; then
    sudo -n jetson_clocks --restore $C/clocks_orig.conf
    sudo -n jetson_clocks --show > $C/show_after.txt 2>&1
    nvpmodel -q > $C/nvpmodel_after.txt 2>&1
    echo "restored $(date +%T)"
  fi
}
trap restore EXIT INT TERM
sudo -n jetson_clocks --show > $C/show_before.txt 2>&1
nvpmodel -q > $C/nvpmodel_before.txt 2>&1
idle dynamic 60
sudo -n jetson_clocks --store $C/clocks_orig.conf
sudo -n jetson_clocks
sleep 5
sudo -n jetson_clocks --show > $C/show_fixed.txt 2>&1
idle fixed 60
bash out/e2e_matrix.sh "" _fixed
echo FINISHED
