function [output] = RunningToTAnalysisByMammal_NewGraphs_For_Yael_Jan2021(PathMain,PathData,PathOutput,Mammal)

fprintf(['** Running ToT Analysis of Mammal: ' Mammal ' **\n']);

%% Pre-Running Code
% Loading Analysis Parameters By Mammal

MammalForAnalysis = Mammal;
    if strcmp(Mammal,'rabbit_PZ_awake')
        MammalForAnalysis = 'rabbit';
    elseif strcmp(Mammal,'mouse_new_anaesthetised')
        MammalForAnalysis = 'mouse';
    end

[AP] = ToTAnalysisParametersByMammal_NewGraphs_For_Yael_Jan2021(Mammal);
FileType = AP.FileType;
Fs = AP.Fs;
ann_ext = AP.ann_ext; % 'qrs' means running peak files
window_minutes = AP.window_minutes;
RRIntervalGraphRange = AP.RRIntervalGraphRange;
% Running Arrangements
[~,~,~] = mkdir([PathOutput filesep 'Graphs' filesep Mammal]);
type_1 = FileType{1};
if length(FileType)>1 type_2 = FileType{2}; end

ToPlot = 'N';
ToPlot_FFT = 'N';
ToRunFullBatch = 'N';

%% Create Data Table
% [T,PeaksCellByTypes_DividedByFs] = CreateTableDataByFolder(PathData,Mammal,FileType,Fs);


%% ECG, Peaks, RR Interval, Histogram Graphs
% if ToPlot == 'Y'
%     CreateECGAnnotatedGraphByFolder(PathData,PathOutput,Mammal,FileType,Fs);
%     CreateRRIntervalGraphByFolder(PathData,PathOutput,Mammal,FileType,Fs);
%     CreateHistogramGraphByFolder(PathData,PathOutput,Mammal,FileType,Fs);
% end % ToPlot

%% Running Batch
fprintf(['** Running Batch_Data Analysis **\n']);
if length(FileType)>1 
    rec_types = {type_1, type_2};
    rec_filenames = {['*' type_1 '*']; ['*' type_2 '*']};
else 
    rec_types = {type_1};
    rec_filenames = {['*' type_1 '*']};
end

% if strcmp('Mammal','rat')

    input_dir = [PathData filesep Mammal filesep 'Peaks Wfdb format'];
    config_file = ['C:\Ido\PZ_new\Data_Analysis\TaleOfTheTail\Code\Config' filesep MammalForAnalysis '_ecg'];
    output_dir = PathMain;
    [~,~,~] = mkdir([output_dir filesep 'Excel_Files' filesep 'By_Window']);
    [~,~,~] = mkdir([output_dir filesep 'Excel_Files' filesep 'All_File']);
    
    params = {config_file,...
    'hrv_freq.power_methods', {'ar', 'welch'},...
    'mse.mse_metrics', true,...
    'mse.mse_max_scale', 20,...
    'dfa.n_incr', 1
    };
    % Mouse Condition <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Ido on
    % mhrv_batch Function !!!!!!!!!! Not the regular function
    batch_data = mhrv.mhrv_batch(input_dir, 'ann_ext', ann_ext, 'writexls', true, 'mhrv_params', params,'window_minutes', window_minutes,...
        'output_dir', [output_dir filesep 'Excel_Files' filesep 'By_Window'] , 'output_filename', Mammal,...
        'rec_filenames', rec_filenames, 'rec_types', rec_types,...
        'min_nn', 0, 'rr_dist_max', Inf, 'rr_dist_min', -Inf);
    
    if ToRunFullBatch == 'Y'
        batch_data_FullRecord = mhrv.mhrv_batch(input_dir, 'ann_ext', ann_ext, 'writexls', true, 'mhrv_params', params,'window_minutes', Inf,...
            'output_dir', [output_dir filesep 'Excel_Files' filesep 'All_File'], 'output_filename', Mammal,...
            'rec_filenames', rec_filenames, 'rec_types', rec_types,...
            'min_nn', 0, 'rr_dist_max', Inf, 'rr_dist_min', -Inf);
    end 

fprintf(['** Done **\n']);

%% RR Interval Full Record
if ToPlot == 'Y'
RRIntervalFullOutputFolder = [PathOutput filesep 'Graphs' filesep Mammal filesep 'RR_Intevral_Graphs_Full_Records'];
[~,~,~] = mkdir(RRIntervalFullOutputFolder);
types_num = rec_types; % batch_data_FullRecord.plot_datas.keys();

for types = 1 : length(types_num)
    files_name = batch_data_FullRecord.plot_datas(types_num{types}).keys(); % check data, inspect plot function - what/how it plots <<<<<<<<<<
    for f_n  = 1 : length(files_name)
        data = batch_data_FullRecord.plot_datas(types_num{types});
        %  []
            % RRinterval.null_vec_X{j,i} = fig{j,i}.Children(2).Children(1).XData;
            % RRinterval.null_vec_Y{j,i} = fig{j,i}.Children(2).Children(1).YData;
        % 'window threshold'
            % RRinterval.window_threshold_X{j,i} = data(files_name{f_n}).filtrr.trr_ma;
            % tresh_low   = plot_data.rri_ma.*(1.0 - plot_data.win_percent/100);
            % thresh_high = plot_data.rri_ma.*(1.0 + plot_data.win_percent/100);
            % plot(ax, plot_data.trr_ma, tresh_low, 'k--', plot_data.trr_ma, thresh_high, 'k--');
        % 'window average'
            RRinterval.window_average_X{types,f_n} = data(files_name{f_n}).filtrr.trr_ma;
            RRinterval.window_average_Y{types,f_n} = data(files_name{f_n}).filtrr.rri_ma;
        % 'Filtered intervals'
            RRinterval.Filtered_intervals_X{types,f_n} = data(files_name{f_n}).filtrr.tnn;
            RRinterval.Filtered_intervals_Y{types,f_n} = data(files_name{f_n}).filtrr.nni;
        % 'RR intervals'
            RRinterval.RR_intervals_X{types,f_n} = data(files_name{f_n}).filtrr.trr;
            RRinterval.RR_intervals_Y{types,f_n} = data(files_name{f_n}).filtrr.rri;
    end
end
        
for i = 1:length(files_name)
    x_axis_start_value = 0;
    hh = figure;
    hold on
        aa = 1; bb = length(types_num);
        if strcmp(types_num{1},'after_trans') bb = 1; aa = length(types_num); end
    for j = aa:-1:bb % 1:length(types_num)
        plot(RRinterval.Filtered_intervals_X{j,i}+x_axis_start_value,RRinterval.Filtered_intervals_Y{j,i});
        x_axis_start_value = RRinterval.Filtered_intervals_X{j,i}(end)+x_axis_start_value;
    end
    ylabel ('RR Intervals (sec)');
    xlabel ('Time (sec)');
    % title(['RR Interval Graph: ' Mammal ' File ' types_num{i}]);
    ylim(RRIntervalGraphRange);
    fig_file_name = [RRIntervalFullOutputFolder filesep files_name{i} '_filtered_full_rr_BasalAndDenervated'];
    savefig(hh, [fig_file_name '.fig'], 'compact');
    saveas(hh, [fig_file_name '.jpg']);
end

end % ToPlot

%% Fourier Transform / Spectrum
if ToPlot_FFT == 'Y'
    fprintf(['** Plotting Spectrum Graphs **\n']);
    output_dir = [PathOutput filesep 'Graphs' filesep Mammal filesep 'Spectrum'];
    [~,~,~] = mkdir(output_dir);
%     r_freq_spectrum(batch_data,'output_dir',output_dir,'output_format','jpeg'); 
    r_freq_spectrum(batch_data,'output_dir',output_dir,'output_format','jpeg','normalize', true); 

    fprintf(['** Done **\n']);
    
end % ToPlot_FFT

% % r_freq_spectrum_united_Ido( batch_data,PathData,PathOutput,Mammal,'normalize', false)
% r_freq_spectrum_united_Ido( batch_data,PathData,PathOutput,Mammal,true)

% Fourier Transform / Spectrum-Histogram
% r_freq_histogram( batch_data,'rec_type_1', {''},'rec_type_2', rec_types, 'output_dir',PathOutput,'output_format','mat')

%% MSE for All Recordings & Each Window
% % % if ToPlot == 'Y'

fprintf(['** Plotting MSE Graphs **\n']);
output_dir = [PathOutput filesep 'Graphs' filesep Mammal filesep 'MSE'];
[~,~,~] = mkdir(output_dir);
% All Files and Types
r_mean_mse_AdditionalGraph_Dec2020(batch_data, 'output_dir', output_dir, 'output_format', 'jpeg');
r_mean_mse_AdditionalGraph_Dec2020(batch_data, 'output_dir', output_dir, 'output_format', 'fig');

FileNames_qrs = dir(fullfile([PathData filesep Mammal filesep 'Peaks Wfdb format'], ['*' FileType{1} '*' '.qrs']));
for i = 1 : length(FileNames_qrs)
    [~, rec_filename, ~] = fileparts(FileNames_qrs(i).name);
    rec_filename_spl = regexpi(rec_filename, type_1, 'split');
    file_name_for_saving = rec_filename_spl{1, 1};
        if isempty(rec_filename_spl{1, 1}) file_name_for_saving = rec_filename; end
    % MSE for each couple of files
    r_mean_mse_AdditionalGraph_Dec2020(batch_data, 'rec_types', rec_types, 'output_dir', output_dir, 'output_format', 'jpeg', 'rec_filename', file_name_for_saving);
    r_mean_mse_AdditionalGraph_Dec2020(batch_data, 'rec_types', rec_types, 'output_dir', output_dir, 'output_format', 'fig',  'rec_filename', file_name_for_saving);
end
fprintf(['** Done **\n']);

% % % end % ToPlot

%% Poincare for Each Window
output_dir = [PathOutput filesep 'Graphs' filesep Mammal filesep 'Poincare' filesep];
[~,~,~] = mkdir([PathOutput filesep 'Graphs' filesep Mammal filesep 'Poincare']);

if ToPlot == 'Y'
fprintf(['** Plotting Poincare Graphs **\n']);

    types_num = rec_types; % batch_data.plot_datas.keys();
    for types = 1 : length(types_num)
        files_name = batch_data.plot_datas(types_num{types}).keys();
        for f_n  = 1 : length(files_name)
            window = sprintf('%d/%d', f_n, length(files_name));
            data = batch_data.plot_datas(types_num{types});
            plot_datas_4type = data(files_name{f_n});
                        
            fig_file_name = [output_dir files_name{f_n} '_poincare'];
            fig_name = sprintf('[%s %s] %s', files_name{f_n}, window, plot_datas_4type.nl.poincare.name);
            f_h = figure('NumberTitle','off', 'Name', fig_name);
            mhrv.plots.plot_poincare_ellipse(gca, plot_datas_4type.nl.poincare);
            savefig(f_h, [fig_file_name '.fig'], 'compact');
            mhrv.util.fig_print(f_h, fig_file_name, 'output_format', 'jpeg', 'font_size', 14, 'width', 20);
            close(f_h);
        end
    end
    
fprintf(['** Done **\n']);
end % ToPlot

%     for i = 1:length(FileType)
%         tmp_data = batch_data.plot_datas(FileType{i});
%         tmp_names = batch_data.plot_datas(FileType{i}).keys();
%         for j = 1:length(tmp_names)
%% Output Data
fprintf(['** Arranging Output Data **\n']);
% Frequency Values
for i = 1:length(FileType)
    tmp = batch_data.stats_tables(FileType{i});
    output.Freq{i} = tmp(1:2,{'VLF_NORM_AR','VLF_NORM_WELCH','LF_NORM_AR','LF_NORM_WELCH','HF_NORM_AR','HF_NORM_WELCH'});
    output.Freq{i} = output.Freq{i}{:,:};
    output.Freq_for_log_graph_mean{i} = tmp(:,{'AVNN','VLF_NORM_WELCH','LF_NORM_WELCH','HF_NORM_WELCH'});
    
    tmp = batch_data.hrv_tables(FileType{i});
    output.Freq_for_log_graph_record{i} = tmp(:,{'AVNN','VLF_NORM_WELCH','LF_NORM_WELCH','HF_NORM_WELCH'});
    
%     if i == 1
%         output.Moran_mat.control = batch_data.hrv_tables(FileType{i});
%         output.Moran_mat.control = output.Moran_mat.control(:,{'MSE1','MSE2','MSE3','MSE4','MSE5','MSE6','MSE7','MSE8','MSE9','MSE10','MSE11','MSE12','MSE13','MSE14','MSE15','MSE16','MSE17','MSE18','MSE19','MSE20'})
%         
%         
%         
%         output.Moran_mat.control = output.Moran_mat.control{:,:};
%     elseif i == 2
%         output.Moran_mat.denervated = batch_data.hrv_tables(FileType{i});
%         output.Moran_mat.denervated = output.Moran_mat.control(:,{'MSE1','MSE2','MSE3','MSE4','MSE5','MSE6','MSE7','MSE8','MSE9','MSE10','MSE11','MSE12','MSE13','MSE14','MSE15','MSE16','MSE17','MSE18','MSE19','MSE20'})
%         output.Moran_mat.denervated = output.Moran_mat.denervated{:,:};
%     end       
end

if length(FileType) == 2
    output.Freq_den_DiviededBy_basal(1,:) = output.Freq{2}(1,:)./output.Freq{1}(1,:);
    output.Freq_den_DiviededBy_basal(2,:) = output.Freq{2}(2,:)./output.Freq{1}(2,:);
else
   vec_length = length(output.Freq{i});
   output.Freq_den_DiviededBy_basal = zeros(2,vec_length); 
end

% RR Intervals for Histogram - Full Record
if ToRunFullBatch == 'Y'
    for i = 1:length(FileType)
        tmp_data = batch_data_FullRecord.plot_datas(FileType{i});
        tmp_names = batch_data_FullRecord.plot_datas(FileType{i}).keys();
        for j = 1:length(tmp_names)
            output.HistogramFilteredRRIntervals_FullRecord{i,j} = tmp_data(tmp_names{j}).filtrr.nni;
            output.HistogramUnfilteredRRIntervals_FullRecord{i,j} = tmp_data(tmp_names{j}).filtrr.rri;
        end
    end
end

% RR Intervals for Histogram - One Window
for i = 1:length(FileType)
    tmp_data = batch_data.plot_datas(FileType{i});
    tmp_names = batch_data.plot_datas(FileType{i}).keys();
    output.HistogramFilteredRRIntervals_names{i} = tmp_names;
    for j = 1:length(tmp_names)
        output.HistogramFilteredRRIntervals{i,j} = tmp_data(tmp_names{j}).filtrr.nni;
        output.HistogramUnfilteredRRIntervals{i,j} = tmp_data(tmp_names{j}).filtrr.rri;
    end
end

% Plot Data Output
    % control
output.plot_datas.data.control.files_name = batch_data.plot_datas(rec_types{1}).keys(); % types_num --> control or denernated
output.plot_datas.data.control.PlotData = batch_data.plot_datas(rec_types{1});

if length(FileType) == 2
    % denervated
output.plot_datas.data.denervated.files_name = batch_data.plot_datas(rec_types{2}).keys(); % types_num --> control or denernated
output.plot_datas.data.denervated.PlotData = batch_data.plot_datas(rec_types{2});
end

% Aranging Output
%output.Table = T;
%output.PeaksCellByTypes = PeaksCellByTypes_DividedByFs;
output.Freq = output.Freq;
output.Freq_order = {'Mean','SE','VLF_NORM_AR','VLF_NORM_WELCH','LF_NORM_AR','LF_NORM_WELCH','HF_NORM_AR','HF_NORM_WELCH'};
output.Freq_for_log_graph_mean = output.Freq_for_log_graph_mean;
output.Freq_for_log_graph_record = output.Freq_for_log_graph_record;

fprintf(['** Done **\n']);

end