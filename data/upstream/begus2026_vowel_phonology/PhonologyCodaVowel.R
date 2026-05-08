#### LOADING LIBRARIES ####

# Loading libraries in
library("readxl")
library("dplyr")
library("tidyverse")
library(diptest)
library(ggplot2)
library(lme4)
library(lmerTest)
library(effects)
library(emmeans)
library(sjPlot)
library(xtable)
library(ggpubr)
library(sjPlot)

# Set working directory

#setwd("")

# Read in data

codas <- read.csv("/codamd.csv") # loads the file with hand-annotated vowels

diphthongs <- read.csv("/codasp.csv")

# Mean frequencies

5/mean(codas[codas$codatype=="5R1",]$Duration)
5/mean(codas[codas$codatype=="1+1+3",]$Duration)
9/mean(codas[codas$codatype=="9i",]$Duration)
8/mean(codas[codas$codatype=="8i",]$Duration)

# Remove codas difficult to transcribe

codas$Vowel<-ifelse(grepl("^a",codas$handv)==TRUE,"a",
                    ifelse(grepl("^i",codas$handv)==TRUE,"i","delete"))


codas<-codas[codas$Vowel!="delete",]

# Create a binary value (a = 0)

codas$VowelBin<-ifelse(codas$Vowel=="a",0,1)



freq <- table(codas$codatype)

keep <- names(freq[freq >= 15])

# Subset the original data frame to keep only those rows with 15 or more coda types
codasBin <- subset(codas, codatype %in% keep)


# Fit the data to a logistic regression linear model

VowelBinModel<-glmer(VowelBin~codatype+(1|whale),data=codasBin,family="binomial")

VowelBinModelSum<-summary(VowelBinModel)
xtable(VowelBinModelSum$coefficients,digits=c(1,2,2,2,4))

# Plot effects

plot(allEffects(VowelBinModel),type="response")

# Plot effects for final figure

VowelBinModel.eff<-allEffects(VowelBinModel)
VowelBinModel.eff.df<-as.data.frame(VowelBinModel.eff)
VowelBinModel.eff.df$codatype
levels(VowelBinModel.eff.df$codatype$codatype)[levels(VowelBinModel.eff.df$codatype$codatype) == "6-NOISE"] <- "6-UNCLASS."


ggplot(VowelBinModel.eff.df$codatype,aes(x=codatype, y=fit))+geom_point()+ geom_errorbar(aes(ymin=lower, ymax=upper), width=0.1)+xlab("Coda Type") +ylab("% i-coda vowel")+ theme_bw() +ggtitle("Proportion of the i-coda vowel across coda types")#+ facet_grid(~feature)





###### Multimodality test

# Only selects a specific type of coda

codas <- subset(codas, codatype == "1+1+3")

table(codas$whale)

(table(codas$whale)["ATWOOD"]+table(codas$whale)["FORK"]+table(codas$whale)["PINCHY"]+table(codas$whale)["TBB"])/length(codas$whale)



vowelIs <- subset(codas, str_detect(Vowel, "^i"))

vowelAs <- subset(codas, str_detect(Vowel, "^a"))




#Dip tests

dip.test(vowelIs[vowelIs$whale=="ATWOOD",]$Duration)
AtwoodI<-dip.test(vowelIs[vowelIs$whale=="ATWOOD",]$Duration)
dip.test(vowelAs[vowelAs$whale=="ATWOOD",]$Duration)
AtwoodA<-dip.test(vowelAs[vowelAs$whale=="ATWOOD",]$Duration)




dip.test(vowelIs[vowelIs$whale=="FORK",]$Duration)
ForkI<-dip.test(vowelIs[vowelIs$whale=="FORK",]$Duration)
dip.test(vowelAs[vowelAs$whale=="FORK",]$Duration)
ForkA<-dip.test(vowelAs[vowelAs$whale=="FORK",]$Duration)

dip.test(vowelIs[vowelIs$whale=="PINCHY",]$Duration)
PinchyI<-dip.test(vowelIs[vowelIs$whale=="PINCHY",]$Duration)
dip.test(vowelAs[vowelAs$whale=="PINCHY",]$Duration)
PinchyA<-dip.test(vowelAs[vowelAs$whale=="PINCHY",]$Duration)

dip.test(vowelIs[vowelIs$whale=="TBB",]$Duration)
TBBI<-dip.test(vowelIs[vowelIs$whale=="TBB",]$Duration)
dip.test(vowelAs[vowelAs$whale=="TBB",]$Duration)
TBBA<-dip.test(vowelAs[vowelAs$whale=="TBB",]$Duration)

Hartigans<-data.frame(Vowel=c(rep("a",4),rep("i",4)),Names=c("Atwood","Fork","Pinchy","TBB","Atwood","Fork","Pinchy","TBB"), D=c(AtwoodA$statistic,ForkA$statistic,PinchyA$statistic,TBBA$statistic,AtwoodI$statistic,ForkI$statistic,PinchyI$statistic,TBBI$statistic), p= c(AtwoodA$p.value,ForkA$p.value,PinchyA$p.value,TBBA$p.value,AtwoodI$p.value,ForkI$p.value,PinchyI$p.value,TBBI$p.value))

print(xtable(Hartigans,digits=c(1,1,2,2,4)),include.rownames = FALSE)




#### Density plots

codas<-codas[codas$whale=="PINCHY"|codas$whale=="FORK"|codas$whale=="ATWOOD"|codas$whale=="TBB",]


#### Linear models

CodasDensityGG<-ggplot(data=codas, aes(x=Duration,fill=Vowel))+geom_density(alpha=.5)+facet_wrap(~whale)+ylab("Density")+theme_minimal()

CodasHistGG<-ggplot(data=codas, aes(x=Duration,fill=Vowel))+geom_histogram(bins=40,position="identity",alpha=0.5)+facet_wrap(~whale)+ylab("Count")+theme_minimal()


mean(codas[codas$Vowel=="a",]$Duration)
mean(codas[codas$Vowel=="i",]$Duration)






#codas<-merge(
#  x = codas, 
#  y = diphthongs[, c("codanum", "autovbycoda")], 
#  by = "codanum", 
#  all.x = TRUE  # this ensures a left join
#)

#codas$VowelAutoBoth<-ifelse(codas$Vowel=="i"&codas$autovbycoda=="i","i",
#                            ifelse(
#                              codas$Vowel=="a"&codas$autovbycoda=="a","a","delete"
#                            ))

#codas<-codas[codas$VowelAutoBoth!="delete",]


#Linear model selection

VowelModel<-lmer(Duration~Vowel+(Vowel|whale),data=codas,REML=F)
VowelModel1<-lmer(Duration~Vowel+(1|whale),data=codas,REML=F)
anova(VowelModel,VowelModel1)

VowelModel<-lmer(Duration~Vowel+(Vowel|whale),data=codas)
VowelModel1<-lmer(Duration~Vowel+(1|whale),data=codas)
AIC(VowelModel,VowelModel1)
summary(VowelModel)
summary(VowelModel1)

#Reported model:  VowelModel (in both models, the signifance of interest is the same)

VowelModelSum<-summary(VowelModel)
xtable(VowelModelSum$coefficients,digits=c(1,2,2,2,2,3))
coef(VowelModel)
plot(allEffects(VowelModel),type="response")

# Extract coefficients

coef(VowelModel)$whale
xtable(coef(VowelModel)$whale)

#Plot random effects

plot_model(VowelModel, type = "re")  +ylim(-.15,.15)+theme_bw()



# Plot effects of the linear model

VowelModel.eff<-allEffects(VowelModel)
VowelModel.eff.df<-as.data.frame(VowelModel.eff)

VowelModel.eff.dfGG<-ggplot(VowelModel.eff.df$Vowel,aes(x=Vowel, y=fit,color=Vowel))+geom_point(aes(shape=Vowel))+ geom_errorbar(aes(ymin=lower, ymax=upper), width=0.1)+ylab("1+1+3 Coda Duration (in s)") +xlab("Coda Vowel")+ theme_bw()#+ facet_grid(~feature)

# Combine plots

ggarrange(CodasHistGG,CodasDensityGG,VowelModel.eff.dfGG,nrow=1,labels = c('a','b','c'))



#### Coarticulation #### 

# Loading the main data frame
coarts <- read.csv("/focal-coarticulation-metadata.csv") # loads the file with hand-annotated vowels

# Eliminate edge cases
coarts <- subset(coarts, deltasec > 0)  

# Only cases with deltasec less than 10
coarts <- subset(coarts, deltasec < 10)

# Choose only four whales

coarts<-coarts[coarts$whale=="PINCHY"|coarts$whale=="FORK"|coarts$whale=="ATWOOD"|coarts$whale=="TBB",]

coarts$match <- ifelse(coarts$handvcat=="a"&coarts$numpks==1,0,
                       ifelse(coarts$handvcat=="i"&coarts$numpks==2,0,
                              ifelse(coarts$handvcat=="a"&coarts$numpks==2,1,
                                     ifelse(coarts$handvcat=="i"&coarts$numpks==1,1,NA)))) 


coarts$type <-ifelse(coarts$coart=="aa","noChange",
                     ifelse(coarts$coart=="ia","Change",
                            ifelse(coarts$coart=="ai","Change",
                                   ifelse(coarts$coart=="ii","noChange",NA))))

# Counts
table(coarts$type)

#Linear Model

coartsModel<-glmer(match~type+handvcat+prevhandvcat+(1|whale),data=coarts,family="binomial")

coartsModelSum<-summary(coartsModel)
coartsModelSum

# Plot the model
plot(allEffects(coartsModel),type="response")

# Extract coefficients
xtable(coartsModelSum$coefficients,digits=c(1,2,2,2,4))


# Plot the model for the final figure
coartsModel.eff<-allEffects(coartsModel)
coartsModel.eff.df<-as.data.frame(coartsModel.eff)

coartsModelgg1<-ggplot(coartsModel.eff.df$type,aes(x=type, y=fit))+geom_point()+geom_line(group=1)+ geom_errorbar(aes(ymin=lower, ymax=upper), width=0.1)+xlab("Type") +ylab("Mismatched first click")+ theme_bw() +ggtitle("Presence of change")#+ylim(0,1)#+ facet_grid(~feature)
coartsModelgg2<-ggplot(coartsModel.eff.df$handvcat,aes(x=handvcat, y=fit))+geom_point()+geom_line(group=1)+ geom_errorbar(aes(ymin=lower, ymax=upper), width=0.1)+xlab("Coda Vowel") +ylab("Mismatched first click")+ theme_bw() +ggtitle("Coda vowel")#+ylim(0,1)#+ facet_grid(~feature)
coartsModelgg3<-ggplot(coartsModel.eff.df$prevhandvcat,aes(x=prevhandvcat, y=fit))+geom_point()+geom_line(group=1)+ geom_errorbar(aes(ymin=lower, ymax=upper), width=0.1)+xlab("Preceding coda vowel") +ylab("Mismatched first click")+ theme_bw() +ggtitle("Preceding coda vowel")#+ylim(0,1)#+ facet_grid(~feature)


# Combine figures
ggarrange(coartsModelgg1,coartsModelgg2,coartsModelgg3,nrow=1,labels = c('a','b','c'))







