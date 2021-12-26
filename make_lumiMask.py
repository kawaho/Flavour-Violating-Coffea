import os, glob, json, uproot
def group_by_run(sorted_run_lumis):
    '''
    Generate a list of lists run-lumi tuples, grouped by run
    Example:
    >>> run_lumis = [(100, 1), (100, 2), (150, 1), (150, 2), (150, 8)]
    >>> list(group_by_run(run_lumis))
    [(100, [1, 2]), (150, [1, 2, 8])]

    '''
    current_run = None
    output = []
    for run, lumi in sorted_run_lumis:
        if current_run is None or run == current_run:
            output.append(lumi)
        else:
            yield (current_run, output)
            output = [lumi]
        current_run = run
    yield (current_run, output)
def collapse_ranges_in_list(xs):
    '''
    Generate a list of contiguous ranges in a list of numbers.
    Example:
    >>> list(collapse_ranges_in_list([1, 2, 3, 5, 8, 9, 10]))
    [[1, 3], [5, 5], [8, 10]]
    '''
    output = []
    for x in xs:
        if not output:
            # Starting new range
            output = [x, x]
        elif x == output[1]+1:
            output[1] = x
        else:
            yield output
            output = [x, x]
    yield output

if __name__ == '__main__':
  years = ['2016preVFP', '2016postVFP', '2017', '2018']
  for year in years:
    samples_era = glob.glob('/hdfs/store/user/kaho/NanoPost_'+year+'_v2/SingleMuon/*')
    sample_paths = {}
    for name in samples_era:
       sample_basename = os.path.basename(name)
       sample_paths[sample_basename] = glob.glob(name+'/*/*/*root')
    for sample_basename in sample_paths:
      samples_names = sample_paths[sample_basename]
      run_lumi_set = []
      runTrees = [i+':LuminosityBlocks' for i in samples_names]
      for runTree in uproot.iterate(runTrees, ['run', 'luminosityBlock'], num_workers=10):
        run_lumi_set.extend(list(zip(runTree['run'].tolist(), runTree['luminosityBlock'].tolist())))
      run_lumis = sorted(run_lumi_set)
      output = {}
      for run, lumis_in_run in group_by_run(run_lumis):
          output[str(run)] = list(collapse_ranges_in_list(lumis_in_run))

      with open(f'json/{year}/{sample_basename}.json', 'w') as f: 
        json.dump(output, f)
        f.close()
  
