for SPKR in {0..35}
  do
    qsub -N spkr_2x2_$SPKR jobscript_2x2.sh $SPKR
  done