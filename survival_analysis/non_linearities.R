x <- seq(from = -40, to = 40, by = 0.01)

sig <- 1 / (1 + exp(-x))
hinge <- function(a) {
  return(max(0, a))
}
relu <- unlist(lapply(x, hinge))

plot(x,
     relu,
     type = 'l',
     ylab = '')
lines(x, sig , col = 'red')
lines(x, tanh(x), col = 'green')
lines(x, abs(tanh(x)), col = 'blue')
