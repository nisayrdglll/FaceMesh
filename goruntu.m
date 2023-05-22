% Haar cascade sınıflandırıcısını yükle
faceDetector = vision.CascadeObjectDetector('haarcascade_frontalface_alt.xml');
eyeDetector = vision.CascadeObjectDetector('haarcascade_eye.xml');

img_path = 'C:\Users\user\Desktop\_\test4.jpg'; % Resim yolunu belirtin
img = imread(img_path); % Resmi oku
gray = rgb2gray(img); % Gri tonlamaya dönüştür

% Yüzleri bul
faces = step(faceDetector, gray);

% Yüzleri işaretle
for i = 1:size(faces, 1)
    x = faces(i, 1);
    y = faces(i, 2);
    w = faces(i, 3);
    h = faces(i, 4);
    img = insertShape(img, 'Rectangle', [x y w h], 'LineWidth', 2, 'Color', [0 255 0]);
    
    roi_gray = gray(y:y+h, x:x+w);
    roi_color = img(y:y+h, x:x+w);
    
    % Gözleri bul
    eyes = step(eyeDetector, roi_gray);
    
    % Gözleri işaretle
    for j = 1:size(eyes, 1)
        x1 = eyes(j, 1);
        y1 = eyes(j, 2);
        w1 = eyes(j, 3);
        h1 = eyes(j, 4);
        roi_color = insertShape(roi_color, 'Rectangle', [x1 y1 w1 h1], 'LineWidth', 2, 'Color', [255 0 0]);
    end
   
end

imshow(img); % Ekranda göster
imwrite(img, 'output_detected.jpg'); % Kaydet
