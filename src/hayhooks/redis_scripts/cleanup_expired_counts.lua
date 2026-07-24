-- hayhooks:cleanup_expired_counts
local ids = redis.call('ZRANGEBYSCORE', KEYS[2], '-inf', ARGV[1], 'LIMIT', 0, ARGV[2])
for _, id in ipairs(ids) do
    if redis.call('ZREM', KEYS[2], id) == 1 then
        local status = redis.call('HGET', KEYS[3], id)
        if status then
            local remaining = redis.call('HINCRBY', KEYS[1], status, -1)
            if remaining < 0 then redis.call('HSET', KEYS[1], status, 0) end
            redis.call('HDEL', KEYS[3], id)
            redis.call('HDEL', KEYS[4], id)
        end
    end
end
return #ids
