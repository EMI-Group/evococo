%% PlatEMO reference benchmark for optimization fidelity
% Runs the 48 algorithms used by EvoCoCo on the selected PlatEMO problems.
% Each (algorithm, problem, seed) run is isolated in a parallel future with
% checkpoint/resume support and a per-run timeout.
%
% Requirements:
%   - PlatEMO on the MATLAB path, or PLATEMO_ROOT set to its root directory
%   - MATLAB Parallel Computing Toolbox
%
% Optional smoke test:
%   Set EVOCOCO_MATLAB_SMOKE_TEST=1 to run AGE-MOEA on DTLZ2 for one seed.
%   Set EVOCOCO_EVALUATION_OUTPUT_DIR to redirect all generated files.

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
timeout_sec = 3 * 3600;
population_size = 100;
generations = 100;
max_fe = population_size * generations;
num_objectives = 3;
seeds = 1:21;

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

problems = {
    'DTLZ1', 'DTLZ2', 'DTLZ3', 'DTLZ4', 'DTLZ5', 'DTLZ6', 'DTLZ7', ...
    'WFG1', 'WFG2', 'WFG3', 'WFG4', 'WFG5', 'WFG6', 'WFG7', 'WFG8', 'WFG9', ...
    'LSMOP1', 'LSMOP2', 'LSMOP3', 'LSMOP4', 'LSMOP5', 'LSMOP6', 'LSMOP7', 'LSMOP8', 'LSMOP9', ...
    'MaF1', 'MaF2', 'MaF3', 'MaF4', 'MaF5', 'MaF6', 'MaF7', 'MaF8', 'MaF9', ...
    'MaF10', 'MaF11', 'MaF12', 'MaF13', 'MaF14', 'MaF15'
};

%% Decision-space dimensions for three objectives
dimension_map = containers.Map();

dimension_map('DTLZ1') = 7;
dimension_map('DTLZ2') = 12;
dimension_map('DTLZ3') = 12;
dimension_map('DTLZ4') = 12;
dimension_map('DTLZ5') = 12;
dimension_map('DTLZ6') = 12;
dimension_map('DTLZ7') = 22;

for index = 1:9
    dimension_map(sprintf('WFG%d', index)) = 12;
    dimension_map(sprintf('LSMOP%d', index)) = 300;
end

for index = 1:6
    dimension_map(sprintf('MaF%d', index)) = 12;
end
dimension_map('MaF7') = 22;
dimension_map('MaF8') = 2;
dimension_map('MaF9') = 2;
dimension_map('MaF10') = 12;
dimension_map('MaF11') = 12;
dimension_map('MaF12') = 12;
dimension_map('MaF13') = 5;
dimension_map('MaF14') = 60;
dimension_map('MaF15') = 60;

%% Optional minimal runtime test (the default remains the full benchmark)
smoke_test = strcmp(getenv('EVOCOCO_MATLAB_SMOKE_TEST'), '1');
if smoke_test
    timeout_sec = 5 * 60;
    population_size = 20;
    generations = 2;
    max_fe = population_size * generations;
    seeds = 1;
    algorithm_names = {'AGE-MOEA'};
    problems = {'DTLZ2'};
    fprintf(['Smoke-test mode: AGE-MOEA on DTLZ2, seed 1, ', ...
             'population %d, generations %d.\n'], population_size, generations);
end

%% Output files
script_dir = fileparts(mfilename('fullpath'));
result_dir = getenv('EVOCOCO_EVALUATION_OUTPUT_DIR');
if isempty(result_dir)
    result_dir = fullfile(script_dir, '..', 'evaluation_results', 'platemo_fidelity');
end
if ~exist(result_dir, 'dir')
    mkdir(result_dir);
end

raw_csv = fullfile(result_dir, 'platemo_trials.csv');
failure_csv = fullfile(result_dir, 'platemo_failures.csv');
summary_csv = fullfile(result_dir, 'platemo_reference.csv');
json_dir = fullfile(result_dir, 'per_generation_jsonl');
if ~exist(json_dir, 'dir')
    mkdir(json_dir);
end

initializeCsv(raw_csv, 'Algorithm,Problem,Seed,IGD,Execution_Time_s\n');
initializeCsv(failure_csv, 'Algorithm,Problem,Seed,Status,Message\n');

fprintf('Raw results: %s\n', raw_csv);
fprintf('Failures:    %s\n', failure_csv);
fprintf('Summary:     %s\n', summary_csv);
fprintf('Generations: %s\n', json_dir);

%% Parallel pool
cluster = parcluster('local');
pool = gcp('nocreate');
if isempty(pool)
    if smoke_test
        pool = parpool('local', 1);
    else
        pool = parpool('local', cluster.NumWorkers);
    end
end
fprintf('Parallel workers: %d\n', pool.NumWorkers);

%% Main benchmark
for algorithm_index = 1:numel(algorithm_names)
    algorithm_name = algorithm_names{algorithm_index};
    algorithm_function = str2func(strrep(algorithm_name, '-', ''));

    for problem_index = 1:numel(problems)
        problem_name = problems{problem_index};
        problem_function = str2func(problem_name);
        dimension = dimension_map(problem_name);

        fprintf('\n========== %s on %s (D=%d) ==========\n', ...
                algorithm_name, problem_name, dimension);

        futures = parallel.FevalFuture.empty(1, 0);
        seed_list = zeros(1, 0);
        submitted_at = zeros(1, 0, 'uint64');

        for seed_index = 1:numel(seeds)
            seed = seeds(seed_index);
            if checkCompleted(raw_csv, algorithm_name, problem_name, seed)
                fprintf('  Skip seed %d (already completed)\n', seed);
                continue;
            end

            futures(end + 1) = parfeval(pool, @runPlatemoTrial, 3, ...
                algorithm_function, problem_function, num_objectives, ...
                dimension, population_size, max_fe, seed, algorithm_name, ...
                problem_name, json_dir);
            seed_list(end + 1) = seed;
            submitted_at(end + 1) = tic;
        end

        run_count = numel(futures);
        if run_count == 0
            fprintf('  All seeds already completed.\n');
            continue;
        end

        pending = true(1, run_count);
        finished = false(1, run_count);
        timed_out = false(1, run_count);
        failed = false(1, run_count);
        runtimes = NaN(1, run_count);
        igd_values = NaN(1, run_count);
        failure_messages = repmat({''}, 1, run_count);
        has_started = false(1, run_count);
        started_at = zeros(1, run_count, 'uint64');

        while any(pending)
            for future_index = find(pending)
                if ~has_started(future_index) && ...
                        strcmp(futures(future_index).State, 'running')
                    has_started(future_index) = true;
                    started_at(future_index) = tic;
                end
                if has_started(future_index) && ...
                        toc(started_at(future_index)) > timeout_sec
                    cancel(futures(future_index));
                    pending(future_index) = false;
                    timed_out(future_index) = true;
                    runtimes(future_index) = toc(submitted_at(future_index));
                    failure_messages{future_index} = 'Per-run timeout exceeded';
                    fprintf('  Seed %d timed out.\n', seed_list(future_index));
                end
            end

            if ~any(pending)
                break;
            end

            try
                pending_indices = find(pending);
                [relative_index, decisions, objectives, constraints] = ...
                    fetchNext(futures(pending), 30);

                if ~isempty(relative_index)
                    future_index = pending_indices(relative_index);
                    pending(future_index) = false;
                    runtimes(future_index) = toc(submitted_at(future_index));

                    try
                        population = SOLUTION(decisions, objectives, constraints);
                        problem = problem_function('M', num_objectives, 'D', dimension);
                        igd_values(future_index) = problem.CalMetric('IGD', population);
                        finished(future_index) = isfinite(igd_values(future_index));
                        failed(future_index) = ~finished(future_index);
                        if failed(future_index)
                            failure_messages{future_index} = 'Non-finite IGD';
                        end
                        fprintf('  Seed %d: %.2fs | IGD %.6f\n', ...
                            seed_list(future_index), runtimes(future_index), ...
                            igd_values(future_index));
                    catch processing_error
                        failed(future_index) = true;
                        failure_messages{future_index} = processing_error.message;
                        fprintf('  Seed %d metric error: %s\n', ...
                            seed_list(future_index), processing_error.message);
                    end
                end
            catch fetch_error
                pending_indices = find(pending);
                for pending_index = pending_indices
                    future = futures(pending_index);
                    if strcmp(future.State, 'finished') && ~isempty(future.Error)
                        pending(pending_index) = false;
                        failed(pending_index) = true;
                        failure_messages{pending_index} = future.Error.message;
                        fprintf('  Seed %d failed: %s\n', ...
                            seed_list(pending_index), future.Error.message);
                    end
                end
                if ~any(strcmp({futures(pending_indices).State}, 'finished'))
                    fprintf('  Waiting: %s\n', fetch_error.message);
                end
            end
        end

        appendResults(raw_csv, failure_csv, algorithm_name, problem_name, ...
            seed_list, igd_values, runtimes, finished, timed_out, failed, ...
            failure_messages);
        writeReferenceSummary(raw_csv, summary_csv, seeds);

        fprintf('  Completed: %d | Timeout: %d | Failed: %d\n', ...
            sum(finished), sum(timed_out), sum(failed));
    end
end

writeReferenceSummary(raw_csv, summary_csv, seeds);

fprintf('\n========== ALL DONE ==========\n');
fprintf('Raw results: %s\n', raw_csv);
fprintf('Failures:    %s\n', failure_csv);
fprintf('Summary:     %s\n', summary_csv);
fprintf('Generations: %s\n', json_dir);

%% Local functions
function [decisions, objectives, constraints] = runPlatemoTrial( ...
        algorithm_function, problem_function, num_objectives, dimension, ...
        population_size, max_fe, seed, algorithm_name, problem_name, json_dir)
    rng(seed, 'twister');
    json_file = generationJsonPath(json_dir, algorithm_name, problem_name, seed);
    if exist(json_file, 'file')
        delete(json_file);
    end
    output_function = @(algorithm, problem) savePerGenerationData( ...
        algorithm, problem, algorithm_name, problem_name, seed, json_dir);
    problem = problem_function( ...
        'M', num_objectives, ...
        'D', dimension, ...
        'N', population_size, ...
        'maxFE', max_fe);
    algorithm = algorithm_function('save', 0, 'outputFcn', output_function);
    algorithm.Solve(problem);
    population = algorithm.result{end};
    decisions = population.decs;
    objectives = population.objs;
    constraints = population.cons;
end

function savePerGenerationData(algorithm, problem, algorithm_name, ...
        problem_name, seed, json_dir)
    if isempty(algorithm.result)
        return;
    end
    population = algorithm.result{end};
    record = struct();
    record.Algorithm = algorithm_name;
    record.Problem = problem_name;
    record.Seed = seed;
    record.FunctionEvaluations = problem.FE;
    record.Generation = ceil(problem.FE / problem.N);
    record.Runtime_s = algorithm.metric.runtime;
    record.IGD = problem.CalMetric('IGD', population);
    record.Objectives = population.objs;

    json_file = generationJsonPath(json_dir, algorithm_name, problem_name, seed);
    file_id = fopen(json_file, 'a');
    if file_id == -1
        error('Could not open per-generation output %s', json_file);
    end
    cleaner = onCleanup(@() fclose(file_id));
    fprintf(file_id, '%s\n', jsonencode(record));
end

function json_file = generationJsonPath(json_dir, algorithm_name, problem_name, seed)
    safe_algorithm = regexprep(algorithm_name, '[^A-Za-z0-9_-]', '_');
    safe_problem = regexprep(problem_name, '[^A-Za-z0-9_-]', '_');
    json_file = fullfile(json_dir, sprintf('%s__%s__seed_%02d.jsonl', ...
        safe_algorithm, safe_problem, seed));
end

function initializeCsv(csv_file, header)
    if exist(csv_file, 'file')
        return;
    end
    file_id = fopen(csv_file, 'w');
    if file_id == -1
        error('Could not create %s', csv_file);
    end
    cleaner = onCleanup(@() fclose(file_id));
    fprintf(file_id, header);
end

function completed = checkCompleted(csv_file, algorithm_name, problem_name, seed)
    completed = false;
    if ~exist(csv_file, 'file')
        return;
    end

    file_id = fopen(csv_file, 'r');
    if file_id == -1
        return;
    end
    cleaner = onCleanup(@() fclose(file_id));
    rows = textscan(file_id, '%s%s%f%f%f', 'Delimiter', ',', 'HeaderLines', 1);
    matches = strcmp(rows{1}, algorithm_name) & ...
              strcmp(rows{2}, problem_name) & ...
              rows{3} == seed & isfinite(rows{4});
    completed = any(matches);
end

function appendResults(raw_csv, failure_csv, algorithm_name, problem_name, ...
        seed_list, igd_values, runtimes, finished, timed_out, failed, messages)
    raw_id = fopen(raw_csv, 'a');
    failure_id = fopen(failure_csv, 'a');
    if raw_id == -1 || failure_id == -1
        if raw_id ~= -1, fclose(raw_id); end
        if failure_id ~= -1, fclose(failure_id); end
        error('Could not open benchmark output files.');
    end
    raw_cleaner = onCleanup(@() fclose(raw_id));
    failure_cleaner = onCleanup(@() fclose(failure_id));

    for index = 1:numel(seed_list)
        seed = seed_list(index);
        if finished(index)
            fprintf(raw_id, '%s,%s,%d,%.12g,%.6f\n', ...
                algorithm_name, problem_name, seed, igd_values(index), runtimes(index));
        elseif timed_out(index)
            fprintf(raw_id, '%s,%s,%d,Inf,%.6f\n', ...
                algorithm_name, problem_name, seed, runtimes(index));
            fprintf(failure_id, '%s,%s,%d,TIMEOUT,"%s"\n', ...
                algorithm_name, problem_name, seed, escapeCsv(messages{index}));
        elseif failed(index)
            fprintf(raw_id, '%s,%s,%d,Inf,%.6f\n', ...
                algorithm_name, problem_name, seed, runtimes(index));
            fprintf(failure_id, '%s,%s,%d,ERROR,"%s"\n', ...
                algorithm_name, problem_name, seed, escapeCsv(messages{index}));
        end
    end
end

function writeReferenceSummary(raw_csv, summary_csv, seeds)
    raw = readtable(raw_csv, 'TextType', 'string');
    algorithms = unique(raw.Algorithm, 'stable');
    problems = unique(raw.Problem, 'stable');

    algorithm_column = strings(0, 1);
    problem_column = strings(0, 1);
    mean_igd_column = zeros(0, 1);
    valid_runs_column = zeros(0, 1);
    expected_runs_column = zeros(0, 1);

    for algorithm_index = 1:numel(algorithms)
        for problem_index = 1:numel(problems)
            mask = raw.Algorithm == algorithms(algorithm_index) & ...
                   raw.Problem == problems(problem_index) & isfinite(raw.IGD);
            values = raw.IGD(mask);
            if isempty(values)
                continue;
            end
            algorithm_column(end + 1, 1) = algorithms(algorithm_index);
            problem_column(end + 1, 1) = problems(problem_index);
            mean_igd_column(end + 1, 1) = mean(values);
            valid_runs_column(end + 1, 1) = numel(values);
            expected_runs_column(end + 1, 1) = numel(seeds);
        end
    end

    reference = table(algorithm_column, problem_column, mean_igd_column, ...
        valid_runs_column, expected_runs_column, ...
        'VariableNames', {'Algorithm', 'Problem', 'MeanIGD', ...
                          'ValidRuns', 'ExpectedRuns'});
    writetable(reference, summary_csv);
end

function escaped = escapeCsv(value)
    escaped = strrep(value, '"', '""');
    escaped = strrep(escaped, sprintf('\r'), ' ');
    escaped = strrep(escaped, sprintf('\n'), ' ');
end
