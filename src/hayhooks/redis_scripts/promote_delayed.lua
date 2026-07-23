-- hayhooks:promote_delayed
local ids = redis.call('ZRANGEBYSCORE', KEYS[1], '-inf', ARGV[1], 'LIMIT', 0, ARGV[2])
for _, id in ipairs(ids) do
    if redis.call('ZREM', KEYS[1], id) == 1 then
        redis.call('XADD', KEYS[2], '*', 'execution_id', id)
    end
end
return ids
