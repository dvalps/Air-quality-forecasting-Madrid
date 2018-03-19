    % THIS SCRIPT READS THE HOURLY DATA OF TEMPERATURE, WIND SPEED, RAINFALL,
    % RELATIVE HUMIDITY and other meteo data FOR THE PERIOD OF 2012 - 2016.
    
    % As with pollution data, the invalid values are turned into 99999.
    
    % CODES:
    % Temperature: 83 --> saved into files: 20XX_temp.mat 
    % Wind speed: 81 --> 20XX_windSpeed.mat 
    % Rainfall: 89 --> saved into files: 20XX_rain.mat 
    % Relative humidity: 86 --> 20XX_hum.mat
    % Wind direction: 82 --> 20XX_windDir.mat
    % Solar radiation: 88 --> 20XX_sol.mat
    % Pressure - 87 --> 20XX_press.mat
    % UV - 80 --> 20XX_uv.mat
    
    clear all
    close all
    
    %The script can be easily adapted to read for another station or parameter
    station = "28079004" %Plaza de España
    %station = "28079056" % Fernandez Ladreda
    %station = "28079024" % Casa de Campo
    
    t = "83"; % the codes of the measured meteo variables
    w = "81";
    r = "89";
    h = "86";
    w_dir = "82"; %wind direction
    sol = "88"; %solar radiation
    press = "87"; %pressure
    uv = "80"; %ultra violet 
    
    parameters = [t; w; r; h; w_dir; sol; press; uv];  % vector of codes of parameters to read
    names_of_parameters = ["temp"; "windSpeed"; "rain"; "hum"; "windDir"; "sol"; "press"; "uv"]
    years = ["2012"; "2013"; "2014"; "2015"; "2016"];
    % Redefine parameters for just wind speed and direction
    %parameters = [w; w_dir];
    %names_of_parameters = ["windSpeed"; "windDir"]
    
    % Files with meteo variables
    fs = ["meteo_2012_2014.txt"; "meteo_2015.txt"; "Meteor Red Vig 16_feb_17.txt"]; %string matrix with file names
    
    for yr=1:length(years)
      year = years(yr, :)
      if year == "2012" || year == "2013" || year == "2014" % Which file to open?
        k = "meteo_2012_2014.txt";
      elseif year  == "2015"
        k = "meteo_2015.txt";
      else k="Meteor Red Vig 16_feb_17.txt";
      endif
      
      for p=1:length(parameters) % for that year, take all the parameters
        parameter = parameters(p,:)

        dat_raw = []; %for reading "raw lines" from the file
        i=1;
        
        f = fopen(k);
        
        % VERY IMPORTANT: For some reason, the code for measuring technique is 98
        % ONLY FOR UV. All other meteo variables have "00" as a code, even though
        % it stands in the pdf it should be 98 as well!
        
        if parameter == "80"
          search_for = strcat(station, parameter, "9802", year(3:4));
        else
          search_for = strcat(station, parameter, "0002", year(3:4));
        endif %if for checking whether the current parameter is uv
        
        while (tline = fgets(f)) != -1
          if char(tline(1:16)) == search_for
            dat_raw(i,:) = tline;
            i = i+1;
          end
        end % end while
        % dat_raw now has the data with each day being one row
        
        fclose(f); % Close it and open it for every parameter because it has to start
                    % reading from the beginning every time

        %Turn it back from double to char
        dat_ch = char(dat_raw);
        
        dat = [];
        dates = [];
        %Take each line (which is a day) and split the data by the characters
        for i=1:size(dat_ch,1)
          dates = [dates; str2double(dat_ch(i,15:20))];
        
          for j=26:6:length(dat_ch(i,:)) %that's where the values start, and 
              if dat_ch(i,j) != 'V'
                %then the last 5 values are not a valid data
                dat_ch(i,j-5:j-1) = '99999';
              end
          end % end for on line
          
          day = strsplit(dat_ch(i,:), {'V', 'N', 'C', 'M', 'Z'});
          first = char(day(1,1));
          first = first(end-4:end); %remove the station code etc. from the first data of the day
          day(1,1) = first;
          
          dat(i,:) = str2double(char(day(1:end-1))); %end-1 becauase the last one is empty cell
        end % end for on the raw data form the file
        
      % Add the dates to the final matrix of NO2 
      dat = [dates dat];
   
      save("-text", strcat(year, "_",names_of_parameters(p,:), ".mat"), "dat");
    end %end for on parameters 
   end % end for on years