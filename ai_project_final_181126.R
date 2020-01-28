# �ΰ����ɰ���
# 2��
# ������ ���翵 �ӿ���

library(plyr)
library(ggplot2)
library(stringr)
library(Matrix)
library(glmnet)
library(xgboost)
library(randomForest)
library(Metrics)
library(dplyr) 
library(caret)
library(scales)
library(e1071)
library(corrplot)

setwd('/data')

train <- read.csv('train.csv', stringsAsFactors = FALSE)
test <- read.csv('test.csv', stringsAsFactors = FALSE)

## N/A �� �ϰ� ó�� train���� saleprice���� 

## ������ �ջ� 
sum(is.na(train$Id))
sum(is.na(train$SalePrice))
sum(is.na(test$Id))

combined.data<- rbind(within(train, rm('Id','SalePrice')), within(test, rm('Id')))
dim(combined.data)

##N/A�� Ȯ��    ##n/a�� �ִ� ���� 34���̴�! 
n.a_column<- which(colSums(is.na(combined.data)) > 0)
length(n.a_column)
sort(colSums(sapply(combined.data[n.a_column], is.na)), decreasing = TRUE)


#0 MSZoning �׸� ����ġ ó�� 

combined.data[is.na(combined.data$MSZoning),c('MSZoning','MSSubClass')]##1916,2217,2251,2905 �� ����ġ

table(combined.data$MSZoning, combined.data$MSSubClass) ##20�� ��� RL�� �е������� ����, 30,70�� RM/�ֺ󰪴�ü  

combined.data$MSZoning[c(2217, 2905)] = 'RL'
combined.data$MSZoning[c(1916, 2251)] = 'RM'

##Ȯ�� - MSZoning ���� �׸� n/a�� 0
sum(is.na(combined.data$MSZoning))


##1 PoolQC(pool quality) ����ġ Ȯ�� 

combined.data[is.na(combined.data$PoolQC),c('PoolQC','PoolArea')]

sum(is.na(combined.data$PoolQC))  ##has 2909 N.A(�ѵ����ʹ� 2919��, ���� 2909���� ����ġ)

##poolarea(����)�� 0�� ���� �������� ���� ��, 0���� ũ�ٸ� �������� ����!  

sum(combined.data$PoolArea>0 & is.na(combined.data$PoolQC)) ## �������� ���������� ǰ���� n/a���� ��=3�� 

## finding the row with above 

combined.data[(combined.data$PoolArea > 0) & is.na(combined.data$PoolQC),c('PoolQC','PoolArea')]
## row 2421, 2504, 2600 have N/A in poolQC which have pool areas 

##pool area ��� poolqc�� ����ġ ä��� 

combined.data[,c('PoolQC','PoolArea')] %>%  ##data�� poolqc, poolarea�� ���� 
  group_by(PoolQC)%>% ##poolQC�� �������� 
  summarise(mean = mean(PoolArea), counts = n())  ## poolarea�� �������

## ex, fa, gd�� �������� ������ ��� ������/ ����ġ �����ϴ� 3���� �˸°� ä���ش�! 

combined.data[(combined.data$PoolArea > 0) & is.na(combined.data$PoolQC),c('PoolQC','PoolArea')]

## 2421�� 368�̹Ƿ� ex�� ������, 2504�� 444�̹Ƿ� ex�� ������. 2600�� 561���� fa�� ����� ������ ����ġ�� none���� ä�� 

combined.data[2421,'PoolQC'] = 'Ex'
combined.data[2504,'PoolQC'] = 'Ex'
combined.data[2600,'PoolQC'] = 'Fa'
combined.data$PoolQC[is.na(combined.data$PoolQC)] = 'None'
##Ȯ���ϱ� 
sum(is.na(combined.data$PoolQC))
sum(is.na(combined.data$PoolArea))


##2 garage���� n/a�� ó�� (GarageYrBlt/GarageFinish/GarageQual/GarageCond/GarageType/GarageArea)

## �� yearbuilt ������ = garage built year�� ������ �� - garageyrblt ó�� XX -> â���� ���� ���̶� N/A���� -ó�� 
length(which(combined.data$GarageYrBlt == combined.data$YearBuilt)) ##2919 obs �� 2216�� ����

na_garageyrblt <- which(is.na(combined.data$GarageYrBlt))

sum(is.na(combined.data$GarageYrBlt)) 
combined.data[is.na(combined.data$GarageYrBlt)==T,]
combined.data[na_garageyrblt, 'GarageYrBlt'] <- 0
##Ȯ���ϱ� 
sum(is.na(combined.data$GarageYrBlt))

## merginge n/a left garage data 
garage.cols <- c('GarageArea', 'GarageCars', 'GarageQual', 'GarageFinish', 'GarageCond', 'GarageType')

#garagecond�� ����ġ Ȯ��
combined.data[is.na(combined.data$GarageCond),garage.cols] ##2127���� garage area, garagecar, type ǥ�� , �׿ܴ� garage�� ���°� 

##2127���� ����� �� ���ϱ� 
a <- which(((combined.data$GarageArea < 370) & (combined.data$GarageArea > 350)) & (combined.data$GarageCars == 1))

sapply(combined.data[a, garage.cols], function(x) sort(table(x))) 
##garagequal:TA   GarageFinish:Unf GarageCond:TA �� ���� ������ ���� 

## 2127���� ä���! 

combined.data[2127,'GarageQual'] = 'TA'
combined.data[2127, 'GarageFinish'] = 'Unf'
combined.data[2127, 'GarageCond'] = 'TA'

##�׿� �� ä���   
###���ڴ� 0���� ä���, N.A�� none���� ä�� -> ���� 0�� ������ �� �� 0���� ä���, N/A�� none���� ä�� 
#####2577�� N.A �� -> none���� ǥ�� 
for (col in garage.cols){
  if (sapply(combined.data[col], is.numeric) == TRUE){
    combined.data[sapply(combined.data[col], is.na), col] = 0
  }
  else{
    combined.data[sapply(combined.data[col], is.na), col] = 'None'
  }
}

##Ȯ�� - garage/pool ���� ��� 0
sort(colSums(sapply(combined.data[n.a_column], is.na)), decreasing = TRUE)

##3kitchenqual / electrical ��� 1�� �� n/a���� ���� - ���� ����� �� ��ü 

combined.data[is.na(combined.data$KitchenQual)==T,] ##1556���� na �� ����  ->�̶� kitchenAbvGr�� 1 
one<-combined.data[combined.data$KitchenAbvGr==1,c("KitchenQual","KitchenAbvGr")]

table(one)  ## TA�� ���� ���⿡ TA�� ��ü 

combined.data$KitchenQual[is.na(combined.data$KitchenQual)] = 'TA'


combined.data[is.na(combined.data$Electrical)==T,] ##1380���� na / �����ִ� ������ ���� ���� �ֺ� ��ü 

str(combined.data$Electrical)
length(which(combined.data$Electrical=='SBrkr')) ##2919�� �� 2671���� sbrkr�̹Ƿ� ��ü 
length(which(combined.data$Electrical=='FuseA'))

combined.data$Electrical[is.na(combined.data$Electrical)] = 'SBrkr'

##Ȯ�� - electrial/ kitchenqual n/a�� 0
sort(colSums(sapply(combined.data[n.a_column], is.na)), decreasing = TRUE)

##4 basement ���� �׸� N/A�� ó���غ���! 

bsmt.column<-c('BsmtExposure','BsmtQual', 'BsmtCond', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2','BsmtFinSF2', 'BsmtUnfSF' ,'TotalBsmtSF', 'BsmtFullBath' ,'BsmtHalfBath')

##bsmtexposure �� ����ġ�� ��
combined.data[is.na(combined.data$BsmtExposure),bsmt.column]

##949, 1488, 2349 row has N/A but others are filled 

B <- which(((combined.data$BsmtQual =='Gd'& combined.data$BsmtCond=='TA')))

sapply(combined.data[B, bsmt.column], function(x) sort(table(x)))###BsmtExposure ��κ� No 

combined.data[c(949, 1488, 2349), 'BsmtExposure'] = 'No' 


##�� �� N/A ���� bsmt�� ���� �� 

###������������ ������, na���� 0 ����, �׿ܴ�none���� ����
for (col in bsmt.column){
  if (sapply(combined.data[col], is.numeric) == TRUE){
    combined.data[sapply(combined.data[col], is.na),col] = 0
  }
  else{
    combined.data[sapply(combined.data[col],is.na),col] = 'None'} } ##2121 row has only N.A

##Ȯ�� - bsmt ���� �׸� n/a�� 0
sort(colSums(sapply(combined.data[n.a_column], is.na)), decreasing = TRUE)

##5exterior1st and Exterior2nd�� n/a���� ���� �ϳ��� �ִ�. 

exterior.column<-c('Exterior1st','Exterior2nd')

combined.data[is.na(combined.data$Exterior1st),exterior.column]
combined.data[is.na(combined.data$Exterior2nd),exterior.column] #2152�� �� �� ��� n/a �̴� 


##exterior ���� �׸��� �����Ƿ� other���� ǥ�� 

combined.data$Exterior1st[is.na(combined.data$Exterior1st)] = 'Other'
combined.data$Exterior2nd[is.na(combined.data$Exterior2nd)] = 'Other'


##Ȯ�� - exteiror ���� �׸� n/a�� 0
sort(colSums(sapply(combined.data[n.a_column], is.na)), decreasing = TRUE)

#6 SaleType, Functional Utilities �׸� N/A ó�� 
##salestype-salescondition�� �������� ó�� 

combined.data[is.na(combined.data$SaleType),'SaleCondition'] ##2490 ���� ����ġ 

table(combined.data$SaleCondition, combined.data$SaleType) ###Salecondition=normal�� ��κ� saletype�� WD(2314��)

combined.data$SaleType[is.na(combined.data$SaleType)] = 'WD'

##Ȯ�� - SaleType ���� �׸� n/a�� 0
sort(colSums(sapply(combined.data[n.a_column], is.na)), decreasing = TRUE)

##functional ���� �׸� 2���� ������ 

combined.data[is.na(combined.data$Functional),]   ##2217, 2474 ���� ������ -�����ִ� �׸����-�ֺ󰪴�ü 

table(combined.data$Functional) 
combined.data$Functional[is.na(combined.data$Functional)] = 'Typ'


##utilities �׸� ������ 2�� 
combined.data[is.na(combined.data$Utilities),] ##1916,1946 �� ����
table(combined.data$Utilities)    #1������ allpub����, �ֺ� ��ü 

## outlier ã�� 
combined.data[(combined.data$Utilities=='NoSeWa'),] #945�� 

combined.data$Utilities[is.na(combined.data$Utilities)] = 'AllPub'
combined.data<-combined.data[-945,]

##Ȯ�� - utilities ���� �׸� n/a�� 0
sort(colSums(sapply(combined.data[n.a_column], is.na)), decreasing = TRUE)




#7 MasVnrArea,MasVnrType ó�� 

## �ϳ��� N/A �����ϸ� ��� :: n/a�� �ִ� ���� ����(?) ���� �� 
combined.data[is.na(combined.data$MasVnrType) | is.na(combined.data$MasVnrArea),c('MasVnrType','MasVnrArea')] ##2611�� area�� ���� 

## N/A ���� �ִ� �� ó�� 
combined.data$MasVnrType[is.na(combined.data$MasVnrType)] = 'None'
combined.data$MasVnrArea[is.na(combined.data$MasVnrArea)] = 0

##2611��ó�� -- area������ ���� ����ġ ä���! 
c<-na.omit(combined.data[,c('MasVnrType','MasVnrArea')]) 

C <- which((c$MasVnrArea < 206) & (c$MasVnrArea > 190)) ##198�� ����� area������ �ֺ� type ã�� 

sapply(combined.data[C, ], function(x) sort(table(x))) ##BrkFace�� 41�� ���� 

combined.data[2611, 'MasVnrType'] = 'BrkFace'


##Ȯ�� - masvnr ���� �׸� n/a�� 0
sort(colSums(sapply(combined.data[n.a_column], is.na)), decreasing = TRUE)

#8 LotFrontage: Linear feet of street connected to property ����ġ 

##����� ���׿� ����� property�� ���� ���̴�. ���׸� �������ڷ�� ó��! 

combined.data['Nbrh.factor'] <- factor(combined.data$Neighborhood, levels = unique(combined.data$Neighborhood))

combined.data$Neighborhood

lot.neighbor <- combined.data[,c('Neighborhood','LotFrontage')] %>%
  group_by(Neighborhood) %>%
  summarise(median = median(LotFrontage,na.rm = TRUE)) ##na �ݵ�� �����Ͽ� �߰��� ���Ѵ� 

lot.neighbor ###�� ������ lotfrontage �߾Ӱ��� ���� 


d = which(is.na(combined.data$LotFrontage)) #lotFrontage�� NA�� ���鿡 ���� 

for (i in d){
  lot.median <- lot.neighbor[lot.neighbor == combined.data$Neighborhood[i],'median']  ##i���� ���׿� ���ٸ�, �߾Ӱ� ��� 
  combined.data[i,'LotFrontage'] <- lot.median[[1]]}  ##lot.median�� ���� ����=�߾Ӱ� ����, lotgrontage�� �߾Ӱ� �ֱ�  

##Ȯ�� - lotgrontage ���� �׸� n/a�� 0
sort(colSums(sapply(combined.data[n.a_column], is.na)), decreasing = TRUE)

#9 Fence: Fence quality N/A ó�� -���� ������ ���� (�׳� fence�� ���� ������ ó��)
str(combined.data$Fence)
table(combined.data$Fence) ##���� �׸��� ����! 

combined.data$Fence[is.na(combined.data$Fence)] = 'None'


##Ȯ�� - fence ���� �׸� n/a�� = �������� ó�� /data���� 
sort(colSums(sapply(combined.data[n.a_column], is.na)), decreasing = TRUE)

#10 MiscFeature ���� �׸� n/a ó��/data������ n/a=none�̶�� ��  

combined.data$MiscFeature[is.na(combined.data$MiscFeature)] = 'None'

##Ȯ�� - MiscFeature ���� �׸� n/a�� = �������� ó�� /data���� 
sort(colSums(sapply(combined.data[n.a_column], is.na)), decreasing = TRUE)


#11 fireplace ����; Fireplaces: Number of fireplaces, FireplaceQu: Fireplace quality -fireplace�� �־�� ǰ���� ����

which((combined.data$Fireplaces > 0) & (is.na(combined.data$FireplaceQu))) ##fireplace�� �ִµ�, qu�� naó���� ���� ����! ��� fireplace�� ���°� 

combined.data$FireplaceQu[is.na(combined.data$FireplaceQu)] = 'None'

##Ȯ�� - FireplaceQu ���� �׸� n/a�� = �������� ó�� /data���� 
sort(colSums(sapply(combined.data[n.a_column], is.na)), decreasing = TRUE)


#12 Alley ���� ; data�������� N/A�� alley ���� ����� ���̶�� �� 

combined.data$Alley[is.na(combined.data$Alley)] = 'None'

##Ȯ�� - Alley ���� �׸� n/a�� = �������� ó�� /data���� 

sort(colSums(sapply(combined.data[n.a_column], is.na)), decreasing = TRUE)

###############N/A ���� Ȯ�� 
sum(is.na(combined.data))

dim(combined.data)








##�� ����ϰ� ���� 




##�� ����ϰ� ���� 

num_features <- names(which(sapply(combined.data, is.numeric)))
cat_features <- names(which(sapply(combined.data, is.character)))

numberic.data <- combined.data[num_features]  #��ġ���ڷḸ�� ������ ���� ������ 


#####����ȭ 


##qual ���� ��� none, po,fa,ta,gd,ex�� ����-> ī�װ����� �з� 
# 11.22 poolqc, BsmtCond �߰�
qual.cols <- c('ExterQual', 'ExterCond', 'GarageQual', 'GarageCond', 'FireplaceQu', 'KitchenQual', 'HeatingQC', 'BsmtQual', 'BsmtCond','PoolQC')

qual.list <- c('None' = 0, 'Po' = 1, 'Fa' = 2, 'TA' = 3, 'Gd' = 4, 'Ex' = 5)

map <- function(cols, list, df){
  for (i in cols){
    df[i] <- as.numeric(list[combined.data[,i]])
  }
  return(df)
}   ##cols�� i �� �������� i���� ��ġ�������� ����Ʈ�� df�� ������ 

numberic.data <- map(qual.cols, qual.list, numberic.data) #��������


####Bsmtexposure ����ȭ 
table(combined.data$BsmtExposure)
group.prices('BsmtExposure')
bsmt.list <- c('None' = 0, 'No' = 1, 'Mn' = 2, 'Av' = 3, 'Gd' = 4)  #bsmt�� ��� �̷��� ���� �ο�
numberic.data <- map(c('BsmtExposure'), bsmt.list, numberic.data)

#####bsmtfintype ����ȭ �������� ���� 
table(combined.data$BsmtFinType1)
table(combined.data$BsmtFinType2)

bsmt.fin.list <- c('None' = 0, 'Unf' = 1, 'LwQ' = 2,'Rec'= 3, 'BLQ' = 4, 'ALQ' = 5, 'GLQ' = 6)
numberic.data <- map(c('BsmtFinType1','BsmtFinType2'), bsmt.fin.list, numberic.data)

## functional ����ȭ 

functional.list <- c('None' = 0, 'Sal' = 1, 'Sev' = 2, 'Maj2' = 3, 'Maj1' = 4, 'Mod' = 5, 'Min2' = 6, 'Min1' = 7, 'Typ'= 8)
numberic.data <- map('Functional', functional.list, numberic.data)

## garage ����ȭ 
table(combined.data$GarageFinish)
garage.fin.list <- c('None' = 0,'Unf' = 1, 'RFn' = 1, 'Fin' = 2)
numberic.data <- map('GarageFinish', garage.fin.list, numberic.data)

## fence ����ȭ
table(combined.data$Fence)

fence.list <- c('None' = 0, 'MnWw' = 1, 'GdWo' = 1, 'MnPrv' = 2, 'GdPrv' = 4)
numberic.data <- map('Fence', fence.list, numberic.data)

# 2918   52

library(plyr)

## MSZoning
table(combined.data$MSZoning)
aggregate(SalePrice~MSZoning, train, mean)
zoning.list <- c('C (all)' = 0, 'RH' = 1, 'FV' = 2, 'RM' = 3, 'RL' = 4)
numberic.data <- map('MSZoning', zoning.list, numberic.data)
sum(is.na(numberic.data$MSZoning))

## LotShape
table(combined.data$LotShape)
aggregate(SalePrice~LotShape, train, mean)
char.list <- c('IR3' = 2, 'IR2' = 3, 'IR1' = 1, 'Reg' = 0)
numberic.data <- map('LotShape', char.list, numberic.data)

## LandContour
table(combined.data$LandContour)
aggregate(SalePrice~LandContour, train, mean)
char.list <- c('Low' = 3, 'Bnk' = 1, 'HLS' = 4, 'Lvl' = 2)
numberic.data <- map('LandContour', char.list, numberic.data)


#drop
col.drops <- c('Utilities')
numberic.data <- numberic.data[,!names(numberic.data) %in% c('Utilities')]

## LotConfig
table(combined.data$LotConfig)
aggregate(SalePrice~LotConfig, train, mean)
char.list <- c('FR3' = 4, 'FR2' = 2, 'CulDSac' = 5, 'Corner' = 3, 'Inside' = 1)
numberic.data <- map('LotConfig', char.list, numberic.data)

## LandSlope
table(combined.data$LandSlope)
aggregate(SalePrice~LandSlope, train, mean)
char.list <- c('Sev' = 3, 'Mod' = 2, 'Gtl' = 1)
numberic.data <- map('LandSlope', char.list, numberic.data)

## Neighborhood  # ���ݺ��� ��������(4) ����(3) ����(2) ����(1) ����(0)
table(combined.data$Neighborhood)
aggregate(SalePrice~Neighborhood, train, mean)
char.list <- c('MeadowV' = 0, 'IDOTRR' = 1, 'Sawyer' = 1, 'BrDale' = 1, 'OldTown' = 1, 'Edwards' = 1, 
               'BrkSide' = 1, 'Blueste' = 1, 'SWISU' = 2, 'NAmes' = 2, 'NPkVill' = 2, 'Mitchel' = 2,
               'SawyerW' = 2, 'Gilbert' = 2, 'NWAmes' = 2, 'Blmngtn' = 2, 'CollgCr' = 2, 'ClearCr' = 3, 
               'Crawfor' = 3, 'Veenker' = 3, 'Somerst' = 3, 'Timber' = 3, 'StoneBr' = 4, 'NoRidge' = 4, 
               'NridgHt' = 4)
numberic.data <- map('Neighborhood', char.list, numberic.data)

## Condition1
table(combined.data$Condition1)
aggregate(SalePrice~Condition1, train, mean)
char.list <- c('RRAe'=1, 'Feedr'=2, 'Artery'=3, 'Norm'=4, 'RRAn'=5,'RRNn'=6, 'RRNe'=7, 'PosN'=8, 'PosA'=9)
numberic.data <- map('Condition1', char.list, numberic.data)

## Condition2
table(combined.data$Condition2)
aggregate(SalePrice~Condition2, train, mean)
char.list <- c('RRAe'=1, 'Feedr'=2, 'Artery'=3, 'Norm'=4, 'RRAn'=5,'RRNn'=6, 'RRNe'=7, 'PosN'=8, 'PosA'=9)
numberic.data <- map('Condition2', char.list, numberic.data)

## BldgType 
table(combined.data$BldgType)
aggregate(SalePrice~BldgType, train, mean)
char.list <- c('1Fam'=5, '2fmCon'=1, 'Duplex'=3,  'Twnhs'=2, 'TwnhsE'=4) 
numberic.data <- map('BldgType', char.list, numberic.data)

## HouseStyle  # ���ݺ��� ��->��(1) 
table(combined.data$HouseStyle)
aggregate(SalePrice~HouseStyle, train, mean)
char.list <- c('1.5Fin'=2, '1.5Unf'=1, '1Story'=3, '2.5Fin'=4, '2.5Unf'=2, '2Story'=4, 'SFoyer'=1, 'SLvl'=3)
numberic.data <- map('HouseStyle', char.list, numberic.data)

## RoofStyle  # ���ݺ� 1~3
table(combined.data$RoofStyle)
aggregate(SalePrice~RoofStyle, train, mean)
char.list <- c('Flat'=3,'Gable'=2, 'Gambrel'=1, 'Hip'=3,  'Mansard'=2, 'Shed'=1)
numberic.data <- map('RoofStyle', char.list, numberic.data)

## RoofMatl # ���ݺ� 1~4
table(combined.data$RoofMatl)
aggregate(SalePrice~RoofMatl, train, mean)
char.list <- c('ClyTile'=1, 'CompShg'=2, 'Membran'=4, 'Metal'=2, 'Roll'=1, 'Tar&Grv'=3, 'WdShake'=3, 'WdShngl'=4)
numberic.data <- map('RoofMatl', char.list, numberic.data)

## Exterior1st # ���ݺ� 1~4
table(combined.data$Exterior1st)
aggregate(SalePrice~Exterior1st, train, mean)
char.list <- c('AsbShng'=1, 'AsphShn'=1, 'BrkComm'=2, 'BrkFace'=3,  'CBlock'=4, 'CemntBd'=4,
               'HdBoard'=2, 'ImStucc'=2, 'MetalSd'=1, 'Other'=4, 'Plywood'=3, 'Stone'=2,  
               'Stucco'=3, 'VinylSd'=3, 'Wd Sdng'=1, 'WdShing'=3) 
numberic.data <- map('Exterior1st', char.list, numberic.data)

## Exterior2nd
table(combined.data$Exterior2nd)
aggregate(SalePrice~Exterior2nd, train, mean)
char.list <- c('AsbShng'=1, 'AsphShn'=1, 'Brk Cmn'=2, 'BrkFace'=3,  'CBlock'=4, 'CmentBd'=4,
               'HdBoard'=2, 'ImStucc'=2, 'MetalSd'=1, 'Other'=4, 'Plywood'=3, 'Stone'=2,  
               'Stucco'=3, 'VinylSd'=3, 'Wd Sdng'=1, 'Wd Shng'=3)
numberic.data <- map('Exterior2nd', char.list, numberic.data)
sum(is.na(numberic.data$Exterior2nd))

## MasVnrType
table(combined.data$MasVnrType)
aggregate(SalePrice~MasVnrType, train, mean)
char.list <- c('BrkCmn'=1, 'BrkFace'=2, 'None'=0, 'Stone'=3) 
numberic.data <- map('MasVnrType', char.list, numberic.data)

## Foundation  # ���ݺ� 1~3
table(combined.data$Foundation)
aggregate(SalePrice~Foundation, train, mean)
char.list <- c('BrkTil'=2, 'CBlock'=2, 'PConc'=3, 'Slab'=1,'Stone'=3,'Wood'=1) 
numberic.data <- map('Foundation', char.list, numberic.data)

## Heating  # ���ݺ� 1~3
table(combined.data$Heating)
aggregate(SalePrice~Heating, train, mean)
char.list <- c('Floor'=2, 'GasA'=2,  'GasW'=3, 'Grav'=1,'OthW'=3, 'Wall'=1)
numberic.data <- map('Heating', char.list, numberic.data)

## CentralAir
table(combined.data$CentralAir)
aggregate(SalePrice~CentralAir, train, mean)
char.list <- c('N'=0,'Y'=1)
numberic.data <- map('CentralAir', char.list, numberic.data)

## Electrical
aggregate(SalePrice~Electrical, train, mean)
char.list <- c('Mix' = 1, 'FuseP' = 2, 'FuseF' = 3, 'FuseA' = 4, 'SBrkr'=5)
numberic.data <- map('Electrical', char.list, numberic.data)

## GarageType
aggregate(SalePrice~GarageType, train, mean)
char.list <- c('None' = 0, 'CarPort' = 1, '2Types' = 2, 'Basment' = 2, 'BuiltIn' = 3, 'Detchd' = 1, 'Attchd' = 3)
numberic.data <- map('GarageType', char.list, numberic.data)
head(numberic.data$GarageType)
sum(is.na(numberic.data$GarageType))

## MiscFeature # ���ݺ� 1~4
aggregate(SalePrice~MiscFeature, train, mean)
char.list <- c('None' = 0, 'TenC' = 4, 'Othr' = 1, 'Gar2' = 3, 'Shed' = 2)
numberic.data <- map('MiscFeature', char.list, numberic.data)

## SaleType # ���ݺ� 1~3
table(combined.data$SaleType)
aggregate(SalePrice~SaleType, train, mean)
char.list <- c('Con' = 3, 'Oth' = 1, 'ConLw' = 1, 'WD'=2, 'COD'=2,
               'ConLI' = 2, 'CWD' = 3, 'ConLD' = 1, 'New'=3)
numberic.data <- map('SaleType', char.list, numberic.data)
sum(is.na(numberic.data$SaleType))

## SaleCondition  # �ٸ� ��� ����
aggregate(SalePrice~SaleCondition, train, mean)
char.list <- c('Abnorml'=1, 'Alloca'=2, 'AdjLand'=3, 'Family'=4, 'Normal'=5, 'Partial'= 0)
numberic.data <- map('SaleCondition', char.list, numberic.data)


#### ONEHOT

## Street 
table(combined.data$Street)
char.list <- c('Pave' = 0, 'Grvl' = 1)
numberic.data <- map('Street', char.list, numberic.data)

## Alley
char.list <- c('None' = 0, 'Pave' = 0, 'Grvl' = 1)
numberic.data <- map('Alley', char.list, numberic.data)

## PavedDrive # what does it mean?
aggregate(SalePrice~PavedDrive, train, mean)
char.list <- c('P' = 0, 'N' = 0, 'Y' = 1)
numberic.data <- map('PavedDrive', char.list, numberic.data)

## MSdwelling ����ȭ

# ������ �ֹ� (�Ʒ�)
numberic.data$MSSubClass <- as.factor(numberic.data$MSSubClass)
numberic.data$MSSubClass <- revalue(numberic.data$MSSubClass, c('20' = 1, '30' = 0, '40' = 0, 
                                                                '45' = 0,'50' = 0, '60' = 1, '70' = 0, '75' = 0, '80' = 0, '85' = 0, '90' = 0, 
                                                                '120' = 1, '150' = 0, '160' = 0, '180' = 0, '190' = 0))
numberic.data$MSSubClass <- as.numeric(gsub("2","0",numberic.data$MSSubClass))
numberic.data$MSSubClass <- as.numeric(numberic.data$MSSubClass)  # 1 �ƴϸ� 2�� �ٲ�
sum(is.na(numberic.data$MSSubClass))

#### CHECK

dim(numberic.data)  # 2918 79 util ���� 78


##������ ��ġ��/ó���� rbind�� train���� �߱⶧���� �̷��� �̾Ƴ��� ���� 

##training data ����
train.data <- cbind(numberic.data[1:1460,], train['SalePrice'])   
head(train.data)  

test.data <- cbind(numberic.data[1461:2919,])   

sum(is.na(numberic.data))



#### Data Preprocessing

all_data <- numberic.data


# transform SalePrice target to log form
train$SalePrice <- log(train$SalePrice + 1)

# for numeric feature with excessive skewness, perform log transformation
# first get data type for each feature
feature_classes <- sapply(names(all_data),function(x){class(all_data[[x]])})
numeric_feats <-names(feature_classes[feature_classes != "character"])

num_features <- names(which(sapply(all_data, is.numeric)))
cat_features <- names(which(sapply(all_data, is.character)))



# get names of categorical features
categorical_feats <- names(feature_classes[feature_classes == "character"])

library(timeDate)
# determine skew for each numeric feature
num_features <- names(which(sapply(all_data, is.numeric)))
skewed_feats <- sapply(num_features, function(x){skewness(all_data[[x]],na.rm=TRUE)})

# keep only features that exceed a threshold for skewness
skewed_feats <- skewed_feats[skewed_feats > 0.75]

# transform excessively skewed features with log(x + 1)
for(x in names(skewed_feats)) {
  all_data[[x]] <- log(all_data[[x]] + 1)
}


# �̰� �����ΰ�? categori ����
col.drops <- c('Utilities')
character.data <- character.data[,!names(numberic.data) %in% c('Utilities')]


# get names of categorical features
categorical_feats <- names(feature_classes[feature_classes == "character"])

dim(train)
dim(test)
dim(all_data)
X_train <- all_data[1:nrow(train),]
X_test <- all_data[(nrow(train)):nrow(all_data),]
#train)+1??
y <- train$SalePrice  #y�� column�� �ƴ϶� ���ڿ�����

X_train$SalePrice<-y

################################################��ó���Ϸ� 

#####jaeyoung modeling#####

colSums(is.na(X_train))

#1modeling-stepwise
#library(leaps)


a<-lm(SalePrice~.,data=X_train)
summary(a)

set.seed(12)

b2<-step(a,direction='both')
summary(b2)

##########rmse ���ϱ� 
install.packages('Metrics')
library(Metrics)

obs <- X_train$SalePrice

both <- predict(b2, newdata=X_train)
#both <- predict(b2, newdata=X_test)
rmse(obs, both)



#########���߰����� Ȯ�� 

library('car')

vif(a)

fit<-lm(SalePrice~.,data=X_train)
summary(fit)

########################
# 2018. 11. 23 Yonghee �߰�
########################
#Before we get into building models, we will holdout the data from train set with sale price to be able to compare observed
#values with predictions. 

####PCA
###For better p-value, we tried to do PCA only with numeric values of the dataset called 'X_train'
##train.data head, dim check
head(X_train)
dim(X_train)

##MSSubClass is not currently numeric value -> so before do cor,  convert into numeric values

X_train$MSSubClass <- as.numeric(X_train$MSSubClass)
X_train$MSSubClass <- as.numeric(gsub("2","0",X_train$MSSubClass))
is.numeric(X_train$MSSubClass)

##Multicollinearity check: check correlation between variables except the 'Saleprice' variable
cor(X_train[1:37])
tr.Saleprice <- y
tr <- X_train[,1:37]

##2. PCA (prcomp) 
aaa<- as.matrix(sapply(tr, as.numeric))
require(graphics)
model <- prcomp(aaa, scale = TRUE)
summary(model)
plot(model$x[,1], model$x[,2])

#####For normalization, log it after +1  
##plot seems to be more scattered than before -> PC1, PC2 to be independent 
aaa_log <- log2(aaa+1 )
aaa_log.obj <- prcomp(aaa_log,scale=TRUE)
plot(aaa_log.obj$x[,1], aaa_log.obj$x[,2]) 

### Biplot: �� ��ü�� ���� ù ��°, �� ��° �ּ��� ���� �� ��ĵ�(biplot) 
biplot(aaa_log.obj, main="Biplot")

#SCREEPLOT 
screeplot(aaa_log.obj, main = "", col = "green", type = "lines",
          pch = 1, npcs = length(aaa_log.obj$sdev))

##Check Cumulative Proportion -> THE SUM FROM PC 1 TO PC26 IS HIGHER THAN 0.80 
##Therefore we should use from PC1 to PC18
summary(aaa_log.obj)


###Linear regression

#Add a training set with principal components 
training.data.pca <- data.frame(aaa_log.obj$x,train$SalePrice)

# Extract first 18 Principal Components 
training.data.pca <- training.data.pca[,1:18] 


# Run a linear regression with PCA transformed data 

dim(training.data.pca)

l.model <- lm(train$SalePrice ~ ., data = training.data.pca)
summary(l.model)

##rmse

# ����3
sqrt(mean(l.model$residuals^2))


########################
# 2018. 11. 26 Sumin
# Ridge & Lasso
########################

##### Pre-processing

# ��ó�� �ٽ�
X_train <- all_data[1:nrow(train),]
X_test <- all_data[(nrow(train)+1):nrow(all_data),]
train$SalePrice <- log(train$SalePrice + 1)  # �߿�
y <- train$SalePrice

# caret ��Ű���� �̿��� model training parameters �����ϱ�
tcr <- trainControl(method="repeatedcv",
                                 number=5,
                                 repeats=5,
                                 verboseIter=FALSE)

# �������� ������ Ȯ��, ������ ���� ������
str(all_data)  # �� num �ƴϸ� int

#### Ridge ȸ�� ��
# alpha�� 0�� ��, Ridge regression

lambdas <- seq(1,0,-0.001)

# train model
set.seed(123)  # for reproducibility
model_ridge <- train(x=X_train, y=y,
                     method="glmnet",
                     preProcess ="medianImpute",
                     metric="RMSE",
                     maximize=FALSE,
                     trControl=tcr,
                     tuneGrid=expand.grid(alpha=0,
                                          lambda=lambdas))


## RMSE �ð�ȭ
## RMSE: ��� ������ ����(Root Mean Square Error; RMSE), ���е�
# ���̷��� ����������???
ggplot(data=filter(model_ridge$result,RMSE<1)) +
  geom_line(aes(x=lambda,y=RMSE))
y <- train$SalePrice

mean(model_ridge$resample$RMSE)  #0.3051705
mean(model_ridge$resample$Rsquared) #0.4179864
summary(model_ridge$resample)

## ���� ��ģ ������ �˰� ���� ��
# extract coefficients for the best performing model
coef <- data.frame(coef.name = dimnames(coef(model_ridge$finalModel,s=model_ridge$bestTune$lambda))[[1]], 
                   coef.value = matrix(coef(model_ridge$finalModel,s=model_ridge$bestTune$lambda)))

coef <- coef[-1,]
picked_features <- nrow(filter(coef,coef.value!=0))
not_picked_features <- nrow(filter(coef,coef.value==0))
cat("Ridge picked",picked_features,"variables and eliminated the other",
    not_picked_features,"variables\n")
coef <- arrange(coef,-coef.value)
imp_coef <- rbind(head(coef,10),
                  tail(coef,10))
ggplot(imp_coef) +
  geom_bar(aes(x=reorder(coef.name,coef.value),y=coef.value),
           stat="identity") +
  ylim(-0.5,0.5) +
  coord_flip() +
  ggtitle("Coefficents in the Ridge Model") +
  theme(axis.title=element_blank())


#### Lasso ȸ�� ��
# alpha�� 1�� ��, Lasso regression

# train model
set.seed(123)  # for reproducibility
model_lasso <- train(x=X_train,y=y,
                     method="glmnet",
                     preProcess ="medianImpute",
                     metric="RMSE",
                     maximize=FALSE,
                     trControl=tcr,
                     tuneGrid=expand.grid(alpha=1,
                                          lambda=c(1,0.1,0.05,0.01,seq(0.009,0.001,-0.001),
                                                   0.00075,0.0005,0.0001)))

## RMSE �ð�ȭ
ggplot(data=filter(model_lasso$result,RMSE<1)) +
  geom_line(aes(x=lambda,y=RMSE))
y <- train$SalePrice

mean(model_lasso$resample$RMSE) #0.3052747
mean(model_lasso$resample$Rsquared) # 0.4174849

## ���� ��ģ ������ �˰� ���� ��
coef <- data.frame(coef.name = dimnames(coef(model_lasso$finalModel,s=model_lasso$bestTune$lambda))[[1]], 
                   coef.value = matrix(coef(model_lasso$finalModel,s=model_lasso$bestTune$lambda)))
coef <- coef[-1,]
picked_features <- nrow(filter(coef,coef.value!=0))
not_picked_features <- nrow(filter(coef,coef.value==0))
cat("Lasso picked",picked_features,"variables and eliminated the other",
    not_picked_features,"variables\n")
coef <- arrange(coef,-coef.value)
imp_coef <- rbind(head(coef,10),
                  tail(coef,10))
ggplot(imp_coef) +
  geom_bar(aes(x=reorder(coef.name,coef.value),y=coef.value),
           stat="identity") +
  ylim(-0.5,0.5) +
  coord_flip() +
  ggtitle("Coefficents in the Lasso Model") +
  theme(axis.title=element_blank())

## id �߰�
ID<-data.frame(test$Id)
testdata<-cbind(ID,X_test)
#?????????????
write.csv(prediction, file = 'predict.csv', row.names = F)


########################

## correlations

# need the SalePrice column
corr.df <- cbind(numberic.data[1:1460,], train['SalePrice'])

# only using the first 1460 rows - training data
correlations <- cor(corr.df)

# only want the columns that show strong correlations with SalePrice
corr.SalePrice <- as.matrix(sort(correlations[,'SalePrice'], decreasing = TRUE))
corr.idx <- names(which(apply(corr.SalePrice, 1, function(x) (x > 0.4 | x < -0.4))))
corrplot(as.matrix(correlations[corr.idx,corr.idx]), type = 'upper', method='color', addCoef.col = 'black', tl.cex = .7,cl.cex = .7, number.cex=.7)


########################

##id �߰� 

ID<-data.frame(test$Id)

#head(X_test)
#head(ID)
testdata<-cbind(ID,X_test)

##prediction

bothpredict <- predict(b2, testdata)
bothpredict <- exp(bothpredict)
head(bothpredict)

rmse(obs, both)

## end 
prediction <- data.frame(ID=testdata$test.Id, SalePrice = bothpredict)

head(prediction)

write.csv(prediction, file = 'predict.csv', row.names = F)