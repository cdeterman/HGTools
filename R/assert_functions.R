


assert_is_in_closed_range <-
  function(x, lower = -Inf, upper = Inf)
  {
    msg <- sprintf("%s is not in range %s to %s.",
                   get_name_in_parent(x), lower, upper)
    assert_engine(x, is_in_closed_range, msg, lower=lower,
                  upper = upper)
  }