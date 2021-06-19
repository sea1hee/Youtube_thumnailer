# YOLO를 활용한 유튜브 썸네일 추천 알고리즘 개선
유튜브에 동영상을 올릴 때 유튜브 자체에서 썸네일을 추천해주는 기능이 존재한다.  
사용하기에 적합하지 않은 썸네일을 추천해주는 것을 문제점으로 생각하고 이를 개선하는 것을 이 프로젝트의 목적으로 두었다.  
http://khuhub.khu.ac.kr/2021-1-capstone-design2/2016104095/edit/master/README.md

아이디어는 다음과 같다.
<br>
<br>


---
## 아이디어 및 사용 오픈소스

위에서 언급한 문제점을 개선하기 위해, 프레임을 의미있게 해주는 가장 중요한 요소를 '객체'라고 가정하였다.  
추천해주는 프레임 내에 무조건 객체가 존재하도록 구현하기 위해 객체 탐지 모델인 [YOLO v3]((https://pjreddie.com/yolo/))를 사용했다.  
동영상을 프레임으로 저장했을 때 화질이 떨어지는 문제점을 조금이나마 해결하기 위해 opencv의 가우시안 블러와 샤프닝을 위한 간단한 커널을 사용하였다.  
프로젝트의 유효성 검증은 머신러닝 YOLO가 배포될 때 학습된 [COCO dataset](https://cocodataset.org)을 사용하였지만, [OIDv4_ToolKit-YOLOv3](https://github.com/EscVM/OIDv4_ToolKit)를 이용해 [Open Image](https://storage.googleapis.com/openimages/web/index.html)에서 데이터셋을 받아 [darknet](https://github.com/pjreddie/darknet) 학습을 진행하였다.


<br>
<br>

---
## 이 프로젝트에서 구현한 내용
1. 한 프레임 내의 유효한 두 물체에 대한 객체 검출  
2. 적절하게 이미지 크롭  
3. Open Image를 활용한 YOLO 학습
<br>
<br>

---

## 프로젝트 실행
프로젝트 파일 내의 code/Project 폴더를 다운받아, 아래와 같이 실행
1. 결과 폴더 비우기
```
python cleanfolder.py
```
2. 입력 동영상 설정
```
code/Project 폴더 내 input 폴더에 example.mp4라는 이름으로 넣기
```
3. 프로젝트 실행 
```
python code/Project/project.py
```

<br>
<br>

---
## 프로젝트 결과 폴더 설명
<br>

#### 1. output1  
검출된 프레임 중 가장 크기가 큰 객체가 존재하는 프레임  
아래는 프레임화된 동영상에서 검출된 bicycle의 크기가 가장 큰 프레임이다  
<br>

<center><img src = "https://user-images.githubusercontent.com/22738293/121802311-9270b800-cc76-11eb-9e0d-aab106dbc47f.png" width="360" height="640"></center>
<br>

#### 2. output2  
검출된 프레임 중 객체가 가장 프레임의 가운데에 위치하는 프레임  
아래는 프레임화된 동영상에서 검출된 bicycle의 크기가 가장 큰 프레임이다 
<br>

<center><img src = "https://user-images.githubusercontent.com/22738293/121802327-addbc300-cc76-11eb-82aa-6562097652a0.png" width="360" height="640"></center>
<br>

#### 3. output3  
가장 가운데 존재하는 객체의 중복 제외 크기의 합이 가장 큰 프레임  
아래는 프레임화된 동영상에서 검출된 bicycle, person의 크기가 가장 큰 프레임이다  
<br>

<center><img src = "https://user-images.githubusercontent.com/22738293/121802343-bcc27580-cc76-11eb-939e-b293b104c99d.png" width="360" height="640"></center>
<br>

#### 4. ouput1_crop, output2_crop, output3_crop  
1~3의 결과를 객체를 중심으로 크롭한 결과이다  
<br>

<center><img src = "https://user-images.githubusercontent.com/22738293/121802373-e11e5200-cc76-11eb-9895-6bcc136d8859.png" width="350" height="200"></center>
<br>

<center><img src = "https://user-images.githubusercontent.com/22738293/121802383-ff844d80-cc76-11eb-95d4-7d33f0241969.png" width="350" height="200"></center>
<br>

<center><img src = "https://user-images.githubusercontent.com/22738293/121802399-0b700f80-cc77-11eb-9b27-5356d1b927d9.png" width="350" height="200"></center>
<br>

---
## 유튜브의 추천 결과
<center><img src = "https://user-images.githubusercontent.com/22738293/121802467-4e31e780-cc77-11eb-939f-83e1bc3848ce.png" width="700" height="400"></center>
<br>

