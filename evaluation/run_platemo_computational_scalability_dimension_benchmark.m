%% PlatEMO computational scalability benchmark: dimension scaling
clc;
clear;

%% Locate PlatEMO
platemo_root = getenv('PLATEMO_ROOT');
if ~isempty(platemo_root)
    addpath(genpath(platemo_root));
end
if isempty(which('platemo'))
    error(['PlatEMO was not found. Add PlatEMO to the MATLAB path or set ', ...
           'the PLATEMO_ROOT environment variable.']);
end

%% Benchmark settings
algorithm_names = {
    'AGE-MOEA', ...
    'BCE-IBEA', ...
    'BCE-MOEA-D', ...
    'BiGE', ...
    'CLIA', ...
    'CMOEA-MS', ...
    'CMOPSO', ...
    'CoMMEA', ...
    'DM-MOEA', ...
    'e-MOEA', ...
    'EFR-RR', ...
    'GDE3', ...
    'GrEA', ...
    'GWASF-GA', ...
    'KnEA', ...
    'LSMOF', ...
    'MaOEA-CSS', ...
    'MOEA-D-AWA', ...
    'MOEA-D-DCWV', ...
    'MOEA-D-DE', ...
    'MOEA-D-DRA', ...
    'MOEA-D-DU', ...
    'MOEA-D-DYTS', ...
    'MOEA-D-FRRMAB', ...
    'MOEA-D-PaS', ...
    'MOEA-D-URAW', ...
    'NSBiDiCo', ...
    'NSGA-II-SDR', ...
    'OSP-NSDE', ...
    'PESA-II', ...
    'PICEA-g', ...
    'PREA', ...
    'S-NSGA-II', ...
    'SIBEA', ...
    'SMPSO', ...
    'SparseEA', ...
    'SparseEA2', ...
    'SPEA-R', ...
    'SSCEA', ...
    't-DEA', ...
    'tDEA-CPBI', ...
    'TELSO', ...
    'TS-NSGA-II', ...
    'TS-SparseEA', ...
    'Two_Arch2', ...
    'VaEA', ...
    'WASF-GA', ...
    'WOF'
};

dim_sizes = [1024, 2048, 4096, 8192, 16384, 32768, 65536];
repeats = 11;
repeat_timeout_seconds = 3 * 60 * 60;
poll_interval_seconds = 30;
skip_larger_after_timeout = true;
pop_size = 1000;
generations = 100;
problemFcn = @DTLZ3;
num_obj = 3;

%% Optional minimal runtime test
smoke_test = strcmp(getenv('EVOCOCO_MATLAB_SMOKE_TEST'), '1');
if smoke_test
    algorithm_names = {'AGE-MOEA'};
    dim_sizes = 12;
    repeats = 1;
    poll_interval_seconds = 1;
    pop_size = 20;
    generations = 2;
    fprintf('Smoke-test mode: AGE-MOEA, DTLZ3, dimension 12.\n');
end
maxFE = pop_size * generations;

%% Output file
script_dir = fileparts(mfilename('fullpath'));
result_dir = getenv('EVOCOCO_EVALUATION_OUTPUT_DIR');
if isempty(result_dir)
    result_dir = fullfile(script_dir, '..', 'evaluation_results', 'platemo_scaling');
end
if ~exist(result_dir, 'dir')
    mkdir(result_dir);
end
csv_file = fullfile(result_dir, 'platemo_dimension_scaling.csv');

%% Parallel pool
if isempty(gcp('nocreate'))
    if smoke_test
        parpool('local', 1);
    else
        parpool;
    end
end

%% Initialize results
if ~exist(csv_file, 'file')
    fid = fopen(csv_file, 'w');
    if fid ~= -1
        fprintf(fid, 'Algorithm,PopSize,Dimension,AvgTime_s,AvgIGD,RawTimes,RawIGDs\n');
        fclose(fid);
    else
        error('Cannot create CSV file.');
    end
end

for algidx = 1:length(algorithm_names)
    algname = algorithm_names{algidx};
    alg = str2func(strrep(algname, '-', ''));

    for didx = 1:length(dim_sizes)
        D = dim_sizes(didx);

        % 断点续跑检查
        skip = false;
        if exist(csv_file, 'file')
            fid = fopen(csv_file, 'r');
            if fid ~= -1
                fileData = textscan(fid, '%s', 'Delimiter', '\n');
                fclose(fid);
                for i = 1:length(fileData{1})
                    line = fileData{1}{i};
                    if startsWith(line, sprintf('%s,%d,%d,', algname, pop_size, D))
                        skip = true;
                        break;
                    end
                end
            end
        end
        if skip
            fprintf('Skipping completed: %s | Dimension = %d\n', algname, D);
            continue;
        end

        [isSupported, unsupportedReason] = isAlgorithmProblemSupported(algname, problemFcn, num_obj, D);
        if ~isSupported
            for next_didx = didx:length(dim_sizes)
                nextD = dim_sizes(next_didx);
                if ~hasCsvPrefix(csv_file, sprintf('%s,%d,%d,', algname, pop_size, nextD))
                    fid = fopen(csv_file, 'a');
                    fprintf(fid, '%s,%d,%d,NaN,NaN,"[UNSUPPORTED_PROBLEM]","[UNSUPPORTED_PROBLEM]"\n', algname, pop_size, nextD);
                    fclose(fid);
                end
            end
            fprintf('Skipping unsupported: %s on %s. %s\n', algname, func2str(problemFcn), unsupportedReason);
            break;
        end

        fprintf('\n--- Running %s with dimension=%d ---\n', algname, D);

        times = zeros(1, repeats);
        igds = zeros(1, repeats);
        success_flags = false(1, repeats);
        timeout_flags = false(1, repeats);
        error_messages = cell(1, repeats);
        processed_flags = false(1, repeats);
        run_start_times = NaT(1, repeats);
        futures = cell(1, repeats);

        for rep = 1:repeats
            futures{rep} = parfeval(@runPlatemoRepeat, 3, alg, problemFcn, num_obj, D, pop_size, maxFE);
        end

        while ~all(processed_flags)
            for rep = find(~processed_flags)
                state = futures{rep}.State;
                if strcmp(state, 'running') && isnat(run_start_times(rep))
                    run_start_times(rep) = datetime("now");
                end

                if strcmp(state, 'finished')
                    try
                        [execTime, finalIGD, errorMsg] = fetchOutputs(futures{rep});
                        if ~isempty(errorMsg)
                            error_messages{rep} = errorMsg;
                        elseif execTime > repeat_timeout_seconds
                            timeout_flags(rep) = true;
                            error_messages{rep} = sprintf('Timed out after %.1f hours', repeat_timeout_seconds / 3600);
                        else
                            times(rep) = execTime;
                            igds(rep) = finalIGD;
                            success_flags(rep) = true;
                            fprintf('  Run %d/%d: %.4fs (IGD: %.4f)\n', rep, repeats, execTime, finalIGD);
                        end
                    catch ME
                        error_messages{rep} = ME.message;
                    end
                    processed_flags(rep) = true;
                elseif strcmp(state, 'failed')
                    error_messages{rep} = futures{rep}.Error.message;
                    processed_flags(rep) = true;
                elseif strcmp(state, 'running') && (seconds(datetime("now") - run_start_times(rep)) >= repeat_timeout_seconds)
                    cancel(futures{rep});
                    timeout_flags(rep) = true;
                    error_messages{rep} = sprintf('Timed out after %.1f hours', repeat_timeout_seconds / 3600);
                    processed_flags(rep) = true;
                end
            end

            if ~all(processed_flags)
                pause(poll_interval_seconds);
            end
        end

        for rep = find(~success_flags)
            if timeout_flags(rep)
                warning('Run %d/%d timed out after %.1f hours.', rep, repeats, repeat_timeout_seconds / 3600);
            else
                warning('Run %d/%d failed: %s', rep, repeats, error_messages{rep});
            end
        end

        success_runs = nnz(success_flags);
        if success_runs > 0 || any(timeout_flags)
            [raw_times_str, raw_igds_str] = formatRawResults(times, igds, success_flags, timeout_flags);

            successful_times = times(success_flags);
            successful_igds = igds(success_flags);
            avg_time = mean(successful_times);
            avg_igd = mean(successful_igds);

            fprintf('  >> Average Time: %.4fs | Average IGD: %.4f\n', avg_time, avg_igd);

            fid = fopen(csv_file, 'a');
            fprintf(fid, '%s,%d,%d,%.4f,%.4f,"%s","%s"\n', algname, pop_size, D, avg_time, avg_igd, raw_times_str, raw_igds_str);
            fclose(fid);
        end

        if any(timeout_flags) && skip_larger_after_timeout
            for next_didx = didx+1:length(dim_sizes)
                nextD = dim_sizes(next_didx);
                if ~hasCsvPrefix(csv_file, sprintf('%s,%d,%d,', algname, pop_size, nextD))
                    fid = fopen(csv_file, 'a');
                    fprintf(fid, '%s,%d,%d,NaN,NaN,"[SKIPPED_AFTER_TIMEOUT]","[SKIPPED_AFTER_TIMEOUT]"\n', algname, pop_size, nextD);
                    fclose(fid);
                end
            end
            fprintf('  >> Timeout reached for %s at dimension=%d. Larger dimensions are skipped.\n', algname, D);
            break;
        end
    end
end

fprintf('\nAll tasks finished. Results saved to: %s\n', csv_file);

function [execTime, finalIGD, errorMsg] = runPlatemoRepeat(alg, problemFcn, num_obj, num_var, pop_size, maxFE)
    execTime = NaN;
    finalIGD = NaN;
    errorMsg = '';
    try
        tStart = tic;
        [decs, objs, cons] = platemo('algorithm', alg, ...
                                     'problem', problemFcn, ...
                                     'M', num_obj, ...
                                     'D', num_var, ...
                                     'N', pop_size, ...
                                     'maxFE', maxFE);
        execTime = toc(tStart);

        population = SOLUTION(decs, objs, cons);
        pro = problemFcn('M', num_obj, 'D', num_var);
        finalIGD = pro.CalMetric('IGD', population);
    catch ME
        errorMsg = ME.message;
    end
end

function [raw_times_str, raw_igds_str] = formatRawResults(times, igds, success_flags, timeout_flags)
    raw_times = cell(1, numel(times));
    raw_igds = cell(1, numel(igds));
    for i = 1:numel(times)
        if success_flags(i)
            raw_times{i} = sprintf('%.4f', times(i));
            raw_igds{i} = sprintf('%.4f', igds(i));
        elseif timeout_flags(i)
            raw_times{i} = 'TIMEOUT';
            raw_igds{i} = 'TIMEOUT';
        else
            raw_times{i} = 'FAILED';
            raw_igds{i} = 'FAILED';
        end
    end
    raw_times_str = sprintf('[%s]', strjoin(raw_times, ', '));
    raw_igds_str = sprintf('[%s]', strjoin(raw_igds, ', '));
end

function matched = hasCsvPrefix(csv_file, prefix)
    matched = false;
    if ~exist(csv_file, 'file')
        return;
    end

    fid = fopen(csv_file, 'r');
    if fid == -1
        return;
    end
    fileData = textscan(fid, '%s', 'Delimiter', '\n');
    fclose(fid);

    for i = 1:length(fileData{1})
        if startsWith(fileData{1}{i}, prefix)
            matched = true;
            return;
        end
    end
end

function [isSupported, reason] = isAlgorithmProblemSupported(algname, problemFcn, num_obj, num_var)
    pro = problemFcn('M', num_obj, 'D', num_var);
    isSupported = true;
    reason = '';

    if any(strcmp(algname, {'LRMOEA', 'RMOEA-DVA'})) && ~any(strcmp(methods(pro), 'Perturb'))
        isSupported = false;
        reason = 'This algorithm requires Problem.Perturb.';
        return;
    end

    if strcmp(algname, 'RMOEA-DVA') && (~isprop(pro, 'delta') || ~isprop(pro, 'H'))
        isSupported = false;
        reason = 'RMOEA-DVA requires problem properties delta and H.';
    end
end

function hasMethod = problemHasMethod(problemFcn, num_obj, num_var, methodName)
    pro = problemFcn('M', num_obj, 'D', num_var);
    hasMethod = any(strcmp(methods(pro), methodName));
end
