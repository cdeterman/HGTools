y <- c(0, 0.0575, 0.1137, 0.2018, 0.3228, 0.5331, 0.8771, 1.0000)
x <- c(0, 0.0040, 0.0105, 0.0258, 0.0540, 0.1210, 0.3419, 1.0000)

y3 <- c(0, rep(1, 7))
y4 <- rep(1, 8)

y2 <- c(0, 0.0540, 0.1155, 0.1588, 0.2440, 0.3910, 0.6519, 1.0000)


gem(x, y, h = .001)
gem(x, y2, h = .001)
gem(x, y3)
gem(x, y4, .001)

s4(seq(0, 1, .01), deriv = 1)

# fit spline to data
s <- splinefun(x, y, method = "monoH.FC")
s2 <- splinefun(x, y2, method = "monoH.FC")
s3 <- splinefun(x, y3, method = "monoH.FC")
s4 <- splinefun(x, y3, method = "monoH.FC")
# get derivatives (i.e. slopes)
s(seq(0,1,.05), deriv = 1)
integrate(s, 0, .3)

plot(c(0,1), c(0,1), type = "l")
#lines(c(0,1), c(0,1))
curve(s(x), 0, 1, col = "green", lwd = 1.5, add=TRUE)
curve(s2(x), 0, 1, col = "red", lwd = 1.5, add = TRUE)
curve(s3(x), 0, 1, col = "blue", lwd = 1.5, add = TRUE)
curve(s4(x), 0, 1, col = "orange", lwd = 1.5, add = TRUE)

gem <- function(x){
    2*(integrate(s, 0, x)$value - .5*(x^2))
}

gem <- function(s, x){
    0.5*(1 + (integrate(s, 0, x)$value - .5*(x^2))/(x - .5*(x^2)))
}


gem <- function(x, y, h = .01){
    # fit spline
    s <- splinefun(x, y, method = "monoH.FC")
    
    # find where slope = 1
    slopes <- data.frame(x = seq(0, 1, h), dx = s(seq(0, 1, h), deriv = 1))
    p <- slopes[which.min(abs(1 - slopes$dx)),]$x
    
    # calculate essentially pAUC
    0.5*(1 + (integrate(s, 0, p)$value - .5*(p^2))/(p - .5*(p^2)))
}



slopes <- data.frame(x = seq(0,1,.01), dx = s(seq(0,1,.01), deriv = 1))
slopes[which.min(abs(1 - slopes$dx)),]

slopes <- data.frame(x = seq(0,1,.01), dx = s2(seq(0,1,.01), deriv = 1))
slopes[which.min(abs(1 - slopes$dx)),]

slopes <- data.frame(x = seq(0,1,.01), dx = s3(seq(0,1,.01), deriv = 1))
slopes[which.min(abs(1 - slopes$dx)),]

gem(s, .3)
gem(s2, .27)
gem(s3, .01)


