for f in results_excitons_2p_1s_P10_1_minus_P20_1_plus_P21_0.5_eps_*
do
  wave_function_analysis.py -m infile.txt -b $f
done

## In one line:
# for f in results_excitons_3Bands_no_couplings_eps_*; do wave_function_analysis.py -m infile.txt -b $f; done
