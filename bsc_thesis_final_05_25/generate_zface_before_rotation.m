function process_db(data_folder)

    function process_folder(folder_name, zface)
        files = dir(folder_name);
        for i = 1:length(files)
            if strcmp(files(i).name, '.') || strcmp(files(i).name, '..')
                continue;
            elseif isdir([folder_name '/' files(i).name])
                process_folder([folder_name '/' files(i).name], zface);
            elseif strcmp(files(i).name(end-2:end), 'png') || strcmp(files(i).name(end-2:end), 'jpg')
                file_name = [folder_name '/' files(i).name]
                img = imread(file_name);
                process_image(img, file_name, zface);
            end
        end
    end

	addpath('./zface');
    addpath('./zface/ZFace_src');
    addpath(genpath('./zface/3rd_party'));
    zface = CZFace('./zface/ZFace_models/zf_ctrl49_mesh512.dat','./zface/haarcascade_frontalface_alt2.xml');
    process_folder(data_folder, zface);

end
