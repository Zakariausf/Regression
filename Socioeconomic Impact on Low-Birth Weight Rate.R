library(caret)
library(car)
library(effects)
library(ggplot2)
library(tidyr)
library(fastDummies)


##Checking data for deciding which transformation method should be performed
str(lbw)
summary(lbw)
##assigning response variable and factor
rate <- lbw$lbwR
gr <- lbw$GrowthR
##subset response variable and factor to keep only numeric predictors
lbws <- lbw[, c(2:3, 5:8)]
lbwg <- gather(lbws,"variable","value")
##R1: Histogram plotting before transforming
ggplot(lbwg,aes(value)) +
  facet_wrap(~ variable,scale="free_x") +
  geom_histogram()
##R1: Histogram plotting after transforming
lbwt <- preProcess(lbws,
                   method=c("BoxCox","center","scale"))
lbwp <- predict(lbwt, lbws)
lbwpg <- gather(lbwp, "variable", "value")
ggplot(lbwpg,aes(value)) +
  facet_wrap(~ variable,scale="free_x") +
  geom_histogram()

##1 Create Dataframe
lbwdf <- data.frame(rate, gr, lbwp)

##2 Linear Regression Model (model1)
model1 <- lm(rate~., lbwdf)
summary(model1)
Anova(model1)

##R2 Effect Plots for all predictors
plot(allEffects(model1))

##3 Variance Inflation factor to check the effects of each predictors in model1
vif(model1)

##R3 check square root of VIF to find out which predictors exceed 2
sqrt_vif <- sqrt(vif(model1))
sqrt_vif

##4 Auxiliary Regression Model to check the predictor with largest square root vif value
amlm <- lm(PropPov~.,lbwdf[-1])
summary(amlm)
Anova(amlm)

##5 Second Regression Model (model2), Hence PropPov has the highest squareroot VIF value, it will be omitted
model2 <- lm(rate~ gr + HInc + PopDen + PropRent + PropBlack + PropHisp, lbwdf)
summary(model2)
Anova(model2)
vif(model2)

##R5 check square root of VIF to find out which predictors exceed 2
sqrt_vif <- sqrt(vif(model2))
sqrt_vif

##R6 Partial F-test
modelnull <- lm(rate~1, lbwdf)
anova(modelnull, model2)

##7 Residual plotting for model 2
residualPlots(model2, pch=16)

#8 Score test of model 2 and each predictor
ncvTest(model2)
ncvTest(model2,~HInc)
ncvTest(model2,~PopDen)
ncvTest(model2,~gr)
ncvTest(model2,~PropRent)
ncvTest(model2,~PropBlack)
ncvTest(model2,~PropHisp)

#9 Influence plot, influence index plot and  dfbetas plot for model2
influencePlot(model2,id=list(n=5,col=2),pch=16)
influenceIndexPlot(model2,id=list(n=5,col=2),pch=16)
dfbetasPlots(model2,id.n=5,pch=16)

#10 Subsetting the observsation with high leverage value and residual plotting
model3 <- lm(rate ~ gr + HInc + PopDen + PropRent + PropBlack + PropHisp, lbwdf[-c(4,43,79,219),])
summary(model3)
Anova(model3)
residualPlots(model3, pch=16)

##11 Model4 with polynomial order for both PropBlack and PropHisp

model4 <- lm(rate~ gr + HInc + PopDen + PropRent + poly(PropBlack, 2) + poly(PropHisp, 2), lbwdf)
summary(model4)
Anova(model4)


##12 Model5 with significant predictors and effect plots

model5 <- lm(rate~ HInc + poly(PropBlack, 2) + poly(PropHisp, 2), lbwdf)
summary(model5)
Anova(model5)
plot(allEffects(model5))

#13 Influence plot for model5
influencePlot(model5,id=list(n=5,col=2),pch=16)

#14 dfbetas plot for model5
dfbetasPlots(model5,id.n=5,pch=16)

##15 Running model 6 (re-run model 4) after subsetting the extreme observations in R 13/14
model6 <- lm(rate~ gr + HInc + PopDen + PropRent + poly(PropBlack, 2) + poly(PropHisp, 2), lbwdf[-c(43),])
summary(model6)
Anova(model6)

##16 Re-run model 5  after subsetting the extreme observations in R 13/14
model7 <- lm(rate~ HInc + poly(PropBlack, 2) + poly(PropHisp, 2), lbwdf[-c(43),])
summary(model7)
Anova(model7)

##17 Model 8 with PopDen (hence it becomes significant in model 6)
model8 <- lm(rate~ HInc + PopDen + poly(PropBlack, 2) + poly(PropHisp, 2), lbwdf[-c(43),])
summary(model8)
Anova(model8)


## 18 K-fold cross-validation
set.seed(555)
myfolds <- createMultiFolds(lbw$lbwR[-43],10,10)
tc <- trainControl(method = "repeatedcv",
                   index = myfolds,
                   repeats=10,
                   verboseIter = TRUE)
model6CV <- train(lbwR ~ poly(PropBlack,2)+PopDen+PropRent+
                    GrowthR + HInc + poly(PropHisp,2), data=lbw[-43,],
                  method="lm",
                  preProcess=c("BoxCox","center","scale"),
                  trControl=tc)
model7CV <- train(lbwR ~ poly(PropBlack,2) +
                    + HInc + poly(PropHisp,2), data=lbw[-c(43),],
                  method="lm",
                  preProcess=c("BoxCox","center","scale"),
                  trControl=tc)
model8CV <- train(lbwR ~ poly(PropBlack,2) + PopDen
                  + HInc + poly(PropHisp,2), data=lbw[-c(43),],
                  method="lm",
                  preProcess=c("BoxCox","center","scale"),
                  trControl=tc)
model6CV
model7CV
model8CV