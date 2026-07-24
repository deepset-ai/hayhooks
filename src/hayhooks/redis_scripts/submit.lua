-- hayhooks:submit
if redis.call('SET', KEYS[1], ARGV[1], 'NX') then
    redis.call('XADD', KEYS[2], '*', 'execution_id', ARGV[2])
    redis.call('HINCRBY', KEYS[3], 'queued', 1)
    redis.call('HSET', KEYS[4], ARGV[2], ARGV[3])
    redis.call('HSET', KEYS[5], ARGV[2], ARGV[4])
    return 1
end
return 0
