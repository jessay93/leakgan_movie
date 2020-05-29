# LeakGAN을 이용한 영화 게시글 제목 생성기  
* 익스트림무비(http://extmovie.com) 내 '영화수다' 게시글 제목과 유사한 한글 문장을 생성합니다.
* generator와 classifier의 구동 환경이 다르므로, Anaconda 등의 **가상환경을 필수로 사용해주시기 바랍니다.**  
* 아래와 같은 3단계로 나누어 정리되었습니다.
	* 데이터 전처리
	* 생성 모델
	* 분류 모델


## 가상 환경 구성
### Environment 1
* **python 3.7+**
* tensorflow:1.14.0-gpu-py3  
* nvidia/cuda:10.0-base-ubuntu16.04  
* **khaiii(한글 형태소분석기)**, wget, pandas 설치  

### Environment 2
* **python 2.7**
* tensorflow:1.14.0-gpu  
* nvidia/cuda:10.0-base-ubuntu16.04   
  
## Data Preprocessing
* **Environment 1 에서 실행**
* 익스트림무비의 [영화수다](https://extmovie.com/movietalk) 게시글 제목 **67만여 건**을 수집, 이 중  700회 이상 등장한 단어로만 이루어진 문장을 선별 (=26,516 문장)
* 각 문장은 카카오 형태소 분석기(khaiii) 로 형태소 분석하였음
* 실행 방법:
	* **경로 : data/**
	* 초기 데이터 파일 : data/movie_highFreq.txt   ,  data/TitleAndView_highFreq.csv
	* 실행 순서 :
		1. python main.py 
		2. python pickleCheck.py (데이터 확인 용도, 필수 X) 
* 예시 :
	* 영화 게시판 제목 데이터(movie_highFreq.txt)
		```shell
		[/SS 겨울왕국/NNG 2/SN ]/SS 영등포/NNP 오/VV 았/EP 는데/EC 무슨/MM 행사/NNG 있/VV 나/EC 보/VX 네요/EC  
		겨울/NNG 왕국/NNG 2/SN 4/SN dx/SL 로/JKB 보러오/VV 았/EP 어요/EC  
		용아맥/NNG 자리/NNG 어디/NP 가/JKS 좋/VA 을까요/EF ??/SF ㅠ/SWK ㅠ/NNG  
		cgv/SL 영화/NNG 관람/NNG 권/XSN 질문/NNG  
		미션/NNG 재/XPN 개봉/NNG 은/JX 취소/NNG 되/XSV ㄴ/ETM 것/NNB 이/VCP ㄴ가요/EF ?/SF
		```

## Generator  
* **Environment 2 에서 실행**
* leakgan 기반의 한글 문장 생성 모델 학습
( 원문 링크 : https://github.com/CR-Gjx/LeakGAN/tree/master/Image%20COCO  )
* 실행 방법:
	* **경로 : generator/**
	* 실행 순서 :
		1. python main.py ( 문장 생성 모델 학습 )
		2. python sentenceGenerate.py --restore=True --model=leakgan-51
		( **문장 생성** , 저장된 leakgan 모델의 epoch 를 참고하여, 위 예시 코드의 "51" 대신 원하는 모델의 epoch 로 변경하여 실행 )
		3. 위에서 생성된 문장 확인 ( 경로 : /data/save_generator )
* 예시 :
	* 한글 문장 생성 결과 ( text_notag_sentenceGenerate.txt )
		```shell
		﻿디즈니 ' 겨울왕국 ' 최신 스틸 공개  
		﻿[ 쥬라기 월드 새 주인공 ]  
		﻿오늘 예매 하 ㄴ 영화 라인업 , TV 예고편  
		﻿' 어 벤 져스 ' 리뷰 이벤트 경품 인증 하 ㅂ니다 ! !  
		﻿날씨 의 아이 스페셜 패키지 상영회 끝나 았 습니다 .
		```
  
## Classifier  
* **Environment 1 에서 실행**
* Attention-RNN 기반의 한글 문장 분류 모델 학습
( 원문 링크 : https://github.com/dongjun-Lee/text-classification-models-tf   )
* 익스트림무비 사이트의 **게시물 제목 & 조회수** 데이터 쌍을 이용하여, generator에서 생성된 문장 중 **조회수**가 높을 것으로 예상되는 게시글 제목을 분류 
* 데이터 & 학습 방법
	* 익스트림무비 게시글 67만 건 중, 조회수 상위 20% 및 하위 20% 내 게시물 제목을 따로 분리하여, 이를 각각 2개의 클래스( ```hit``` / ```non-hit``` ) 로 labeling 후 저장 
	( ```hit``` / ```non-hit``` 데이터 : 각각 약 12만 문장으로 구성  )
	* 해당 데이터를 이용하여, Attention-RNN 기반 한글 문장 분류 모델 학습
* 실행 방법:
	* **경로 : classifier/**
	* 초기 데이터 파일 : data/train.csv , data/test.csv
	* 실행 순서 :
		1. python train.py ( 문장 분류 모델 학습 )
		2. python test_gen.py		( **generator** 에서 생성된 문장 별 조회수 예측 )
		3. 위에서 분류된 문장 별 조회수 확인 ( 경로 : /data/save_classifier )
* 예시 :
	* 한글 문장 분류 결과 ( classi_Result.txt )
		```shell
		[ 분노 의 질주 7 ] 19 주년 기념 포스터 100.0
		설국열차 재 개봉 언제 하 죠 ?  99.63
		...
		cgv 질문 . 있 늦 네요 .... ! !    0.09
		날씨 의 아이 2 회 차 관람 하 러 오 았 습니다 .  0.04
		```

## 기타
* 학습 시간 : 
	* CPU : Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz
	* GPU : GeForce RTX 2060 6GB
		* generator : pre-train(<1.5h), adversarial-train(<5h)
		* classifier :  ~= 10분
