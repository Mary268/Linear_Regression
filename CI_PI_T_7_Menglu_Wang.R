
data_case3 = read.csv("/Users/marywang/Documents/MSCI_719/719_case3/VUMC_Schedule.csv")
actual = data_case3$Actual
T7 = data_case3$T...7
LR_case3 = lm(actual~T7, data = data_case3)
#y = 7.0326+0.0475*160
#y_CI = predict(LR_case3, newdata=data.frame(x=x), interval="confidence", level=0.95)
#y_PI = predict(LR_case3, newdata=data.frame(x=x), interval="prediction", level=0.95)
# plot CI and PI lines
x_rg = range(0,120)
x_rg_expand = data.frame(T7=seq(x_rg[1],x_rg[2],length.out=12))
CI_y_bar  = predict(LR_case3, newdata=x_rg_expand, interval="confidence", level=0.99) #CI of y_mean
PI_y_zero = predict(LR_case3, newdata=x_rg_expand, interval="prediction", level=0.99) #PI of y_zero
plot(T7, actual, main = 'CI and PI lines of T - 7 SLR')
abline(a = LR_case3$coefficients[1], b = LR_case3$coefficients[2], col = 'blue', lwd=2)
lines(x_rg_expand$T7,CI_y_bar[,2],lty=2,col="red",lwd=2)  #lwr of CI y_mean
lines(x_rg_expand$T7,CI_y_bar[,3],lty=2,col="red",lwd=2)  #upr of CI y_mean
lines(x_rg_expand$T7,PI_y_zero[,2],lty=4,col="green",lwd=2)  #lwr of PI y_zero
lines(x_rg_expand$T7,PI_y_zero[,3],lty=4,col="green",lwd=2)  #upr of PI y_zero
legend("topleft",c("CI","PI"),lty=2:4,cex=2,
       lwd=rep(2,3),col=c("red","green"),bty="n")