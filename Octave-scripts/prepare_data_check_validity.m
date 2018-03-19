    % THIS SCRIPT READS THE HOURLY DATA OF dat AT THE STATION OF PZA. DE ESPAÑA
    % THE ONLY THING THAT NEEDS TO BE CHANGED IS THE NAME OF THE FILE TO READ
    % THE DATA FROM, ACCORDINGLY TO THE MONTH THAT IS WANTED
    
    %This script also checks for invalid values. 
    %ALL THE INVALID VALUES ARE SUBSTITUTED WITH 99999. THAT WAY,
    %WHEN I SEE 99999 IN MY DATA SET, IT MEANS IT WAS AN INVALID VALUE.
  
    
    %For example, for Pza de España, what I am searching for in the first 10 characters is: 2807900408
    %Therefore, all the lines that I read from a file and that do not start with
    %those 10 characters will be discarded.
    
    clear all
    close all
    
    %The script can be easily adapted to read for another station or parameter
    station = "28079004" %Plaza de España
    station = "28079056" % Fernandez Ladreda
    station = "28079024" % Casa de Campo: it has SO2, CO, NO, dat, O3, PM10, PM2.5
    
    no2 = "08" % no2
    so2 = "01" %SO2
    co = "06" %CO
    no = "07" %NO
    pm25 = "09" %pm2.5
    pm10 = "10" %pm10
    nox = "12" %NOx
    o3 = "14" %O3
    accidrain = "92" %accid rain
    # Other possible: sound, hydrocarbons, benzene
    
    parameters = [no2; so2; co; no; pm25; pm10; o3]; #for casa de campo
    names_of_parameters = ["no2"; "so2"; "co"; "no"; "pm25"; "pm10"; "o3"];
    % This is a cell string - access a specific string by:
    % parameters(1,:) = 08 , while parameters(1,2) = 8
    
    %HARDCODED FOR NOW - that way it is cronollogically sorted
    months12 = ['2012/ene_mo12.txt'; '2012/feb_mo12.txt'; '2012/mar_mo12.txt'; '2012/abr_mo12.txt';'2012/may_mo12.txt'; '2012/jun_mo12.txt';'2012/jul_mo12.txt'; '2012/ago_mo12.txt';'2012/sep_mo12.txt'; '2012/oct_mo12.txt'; '2012/nov_mo12.txt'; '2012/dic_mo12.txt' ]; %string matrix with file names in 2012
    months13 = ['2013/ene_mo13.txt'; '2013/feb_mo13.txt'; '2013/mar_mo13.txt'; '2013/abr_mo13.txt';'2013/may_mo13.txt'; '2013/jun_mo13.txt';'2013/jul_mo13.txt'; '2013/ago_mo13.txt';'2013/sep_mo13.txt'; '2013/oct_mo13.txt'; '2013/nov_mo13.txt'; '2013/dic_mo13.txt' ]; %string matrix with file names in 2013
    months14 = ['2014/ene_mo14.txt'; '2014/feb_mo14.txt'; '2014/mar_mo14.txt'; '2014/abr_mo14.txt';'2014/may_mo14.txt'; '2014/jun_mo14.txt';'2014/jul_mo14.txt'; '2014/ago_mo14.txt';'2014/sep_mo14.txt'; '2014/oct_mo14.txt'; '2014/nov_mo14.txt'; '2014/dic_mo14.txt' ]; %string matrix with file names in 2014
    months15 = ['2015/ene_mo15.txt'; '2015/feb_mo15.txt'; '2015/mar_mo15.txt'; '2015/abr_mo15.txt';'2015/may_mo15.txt'; '2015/jun_mo15.txt';'2015/jul_mo15.txt'; '2015/ago_mo15.txt';'2015/sep_mo15.txt'; '2015/oct_mo15.txt'; '2015/nov_mo15.txt'; '2015/dic_mo15.txt' ]; %string matrix with file names in 2015
    months16 = ['2016/ene_mo16.txt'; '2016/feb_mo16.txt'; '2016/mar_mo16.txt'; '2016/abr_mo16.txt';'2016/may_mo16.txt'; '2016/jun_mo16.txt';'2016/jul_mo16.txt'; '2016/ago_mo16.txt';'2016/sep_mo16.txt'; '2016/oct_mo16.txt'; '2016/nov_mo16.txt'; '2016/dic_mo16.txt' ]; %string matrix with file names in 2016
    
    years = [months12; months13; months14; months15; months16];
    
    % length(years) = 60
    % access with years(idx,:), with 2012 having indices 1-12, 2013 indices 13-24, ..., 2016: 49 - 60
    
    
    for yr = 1:12:length(years)
    month = years(yr:yr+11, :)  # all the 12 months in the year
    
      for p=1:length(parameters) # for that year, take all the parameters
        parameter = parameters(p,:)
        
        data_final = [];
        dates = [];
        for i=1:length(month(:,1))
            f = fopen(month(i,:));  % file's name is in month(i,:)

            data_raw = []; %for reading "raw lines" from the file
            i = 1;
            
            while (tline = fgets(f)) != -1
              if char(tline(1:10)) == strcat(station, parameter)
                data_raw(i,:) = tline;
                i = i+1;
              end %end if
            end %end while
            
            
            fclose(f);

            %Turn it back from double to char
            data_ch = char(data_raw);
            
            dat = [];
            %Take each line (which is a day) and split the data by the characters
            for i=1:size(data_ch,1)
              
              % At the beginning of each row, extract also a date - 6 charactes,
              % on indices from 15 to 20. For example, April 1st of 2015 is: 150401
              
              % Put it in dates, and later just append to the matrix, but as a 6-digit number!
              dates = [dates; str2double(data_ch(i,15:20))];
            
              for j=26:6:length(data_ch(i,:)) %that's where the values start
                  if data_ch(i,:) != 'V'
                    %then the last 5 values are not a valid data
                    data_ch(i,j-5:j-1) = '99999';
                  end
              end   %end for
              
              day = strsplit(data_ch(i,:), {'V', 'N', 'C', 'M', 'Z'});
              first = char(day(1,1));
              first = first(end-4:end); %remove the station code etc. from the first data of the day
              day(1,1) = first;
              
              dat(i,:) = str2double(char(day(1:end-1))); %end-1 becauase the last one is empty cell
            end %end for on data_ch
            
            %The final product is dat hourly data. For each day, I have 24 values of data.
            
            data_final = [data_final; dat]; %add the current month into the data_final matrix for the whole year
       end %end for on the months of the year
   
        % Add the dates to the final matrix of dat 
        data_final = [dates data_final];
        % Save the file
        %save strcat(month(1,1:4), "_",names_of_parameters(1,:), ".mat") data_final
        save("-text", strcat(month(1,1:4), "_",names_of_parameters(p,:), ".mat"), "data_final");
     end % for loop on parameters
  
  end % for loop on years