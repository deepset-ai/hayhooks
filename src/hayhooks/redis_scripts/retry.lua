-- hayhooks:retry
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return 0 end
local current_payload = redis.call('GET', KEYS[2])
local current = current_payload and cjson.decode(current_payload) or {}
local canceled = current['cancel_requested_at'] ~= nil and current['cancel_requested_at'] ~= cjson.null
local new_status
local candidate
if canceled then
    candidate = cjson.decode(ARGV[3])
    candidate['cancel_requested_at'] = current['cancel_requested_at']
    candidate['cancel_reason'] = current['cancel_reason']
    candidate['bounded_progress'] = current['bounded_progress']
    candidate['sequence'] = math.max(candidate['sequence'] or 0, current['sequence'] or 0) + 1
    new_status = candidate['status']
    redis.call('SET', KEYS[2], cjson.encode(candidate), 'EX', ARGV[4])
    redis.call('ZADD', KEYS[6], ARGV[9], ARGV[6])
    redis.call('HSET', KEYS[7], ARGV[6], new_status)
else
    candidate = cjson.decode(ARGV[2])
    new_status = candidate['status']
    redis.call('SET', KEYS[2], ARGV[2])
    redis.call('ZADD', KEYS[3], ARGV[5], ARGV[6])
end
if current['status'] ~= new_status then
    local remaining = redis.call('HINCRBY', KEYS[5], current['status'], -1)
    if remaining < 0 then redis.call('HSET', KEYS[5], current['status'], 0) end
    redis.call('HINCRBY', KEYS[5], new_status, 1)
end
redis.call('XACK', KEYS[4], ARGV[7], ARGV[8])
redis.call('XDEL', KEYS[4], ARGV[8])
redis.call('DEL', KEYS[1])
if canceled then
    redis.call('HDEL', KEYS[8], ARGV[6])
else
    redis.call('HSET', KEYS[8], ARGV[6], candidate['definition_revision'])
end
redis.call('HSET', KEYS[9], ARGV[6], candidate['sequence'] or 0)
return canceled and 2 or 1
