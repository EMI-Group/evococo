function result_dir = prepare_platemo_fidelity_output( ...
        script_dir, result_dir, smoke_test, population_size, generations, num_objectives)
% Separate smoke results and reject resume from a different or unknown budget.
    if isempty(result_dir)
        directory_name = 'platemo_fidelity';
        if smoke_test
            directory_name = 'platemo_fidelity_smoke';
        end
        result_dir = fullfile(script_dir, '..', 'evaluation_results', directory_name);
    end
    if ~exist(result_dir, 'dir')
        mkdir(result_dir);
    end

    protocol = struct('schema_version', 1, ...
                      'population_size', population_size, ...
                      'generations', generations, ...
                      'objectives', num_objectives, ...
                      'smoke_test', logical(smoke_test));
    protocol_file = fullfile(result_dir, 'benchmark_protocol.json');
    if exist(protocol_file, 'file')
        previous = jsondecode(fileread(protocol_file));
        if ~isstruct(previous) || ~isequal(orderfields(previous), orderfields(protocol))
            error('EvoCoCo:ProtocolMismatch', ...
                  ['The output directory contains a different benchmark protocol. ', ...
                   'Choose a new EVOCOCO_EVALUATION_OUTPUT_DIR: %s'], result_dir);
        end
        return;
    end

    existing_outputs = {'platemo_trials.csv', 'platemo_reference.csv', ...
                        'platemo_failures.csv'};
    for index = 1:numel(existing_outputs)
        if exist(fullfile(result_dir, existing_outputs{index}), 'file')
            error('EvoCoCo:LegacyResults', ...
                  ['Existing results have no benchmark protocol and cannot be ', ...
                   'resumed safely. Preserve them and choose a new ', ...
                   'EVOCOCO_EVALUATION_OUTPUT_DIR: %s'], result_dir);
        end
    end

    temporary_file = tempname(result_dir);
    file_id = fopen(temporary_file, 'w');
    if file_id == -1
        error('Could not create protocol metadata in %s', result_dir);
    end
    cleaner = onCleanup(@() fclose(file_id));
    fprintf(file_id, '%s\n', jsonencode(protocol));
    clear cleaner;
    [moved, message] = movefile(temporary_file, protocol_file, 'f');
    if ~moved
        error('Could not save benchmark protocol: %s', message);
    end
end
