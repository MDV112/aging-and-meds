function [] = addending2rr()
%% This function takes the peaks of aging mice (Ismayil found in C:\Users\smorandv.STAFF\Documents\PhD\Aging Data\)
% , converts them to rr, divdes them to windows and adds id, age and med as
% labels. It sends the results back to aging and meds
%%
close all
clear
clc
%%
old_dir = pwd;
h = 1;
for age = 6:3:30
    for med = 0:1
        if med == 0 % (control)
            curr_data = ['C:\Users\smorandv.STAFF\Documents\PhD\Aging Data\C57 '...
                    num2str(age) 'm mat\ecg_control\peaks_control'];
        else % (after trans)
            curr_data = ['C:\Users\smorandv.STAFF\Documents\PhD\Aging Data\C57 '...
                    num2str(age) 'm mat\ecg_after_trans\peaks_after_trans'];
        end
        cd(curr_data)
        listing = dir;
        for i = 1:length(listing)
            curr_mouse = listing(i).name;
            if contains(curr_mouse,'.mat') % check that it is a mat file 
                load(curr_mouse, 'Data', 'Fs')
                rr = diff(Data)/Fs;
                % look for the id (tag)
                for j = 1:length(curr_mouse)
                    idx1 = strfind(curr_mouse, '(');
                    idx2 = strfind(curr_mouse, ')');
                    id = str2double(curr_mouse(1 + idx1:idx2 - 1));
                end
                labeled_rr = zeros(length(rr) + 3,1);
                labeled_rr(1:end-3) = rr;
                labeled_rr(end-2) = id;
                labeled_rr(end-1) = age;
                labeled_rr(end) = med;
                save([old_dir '\aging\' num2str(h) '.mat'], 'labeled_rr')
                h = h + 1;
                fprintf('saving %d/%d of age %d with med %d \n',i,length(listing),age,med)
            end
        end
    end
end
end