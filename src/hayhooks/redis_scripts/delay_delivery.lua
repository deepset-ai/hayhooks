-- hayhooks:delay_delivery
redis.call('ZADD', KEYS[1], ARGV[1], ARGV[2])
redis.call('XACK', KEYS[2], ARGV[3], ARGV[4])
redis.call('XDEL', KEYS[2], ARGV[4])
return 1
