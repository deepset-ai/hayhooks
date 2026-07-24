-- hayhooks:complete
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return -1 end
local payload = ARGV[2]
local current_payload = redis.call('GET', KEYS[2])
local current = current_payload and cjson.decode(current_payload) or {}
local canceled = current['cancel_requested_at'] ~= nil and current['cancel_requested_at'] ~= cjson.null
if canceled then
    local candidate = cjson.decode(ARGV[3])
    candidate['cancel_requested_at'] = current['cancel_requested_at']
    candidate['cancel_reason'] = current['cancel_reason']
    candidate['bounded_progress'] = current['bounded_progress']
    candidate['sequence'] = math.max(candidate['sequence'] or 0, current['sequence'] or 0) + 1
    payload = cjson.encode(candidate)
end
local final = cjson.decode(payload)
local old_status = current['status']
local new_status = final['status']
if old_status ~= new_status then
    local remaining = redis.call('HINCRBY', KEYS[4], old_status, -1)
    if remaining < 0 then redis.call('HSET', KEYS[4], old_status, 0) end
    redis.call('HINCRBY', KEYS[4], new_status, 1)
end
redis.call('ZADD', KEYS[5], ARGV[8], ARGV[7])
redis.call('HSET', KEYS[6], ARGV[7], new_status)
redis.call('SET', KEYS[2], payload, 'EX', ARGV[4])
redis.call('XACK', KEYS[3], ARGV[5], ARGV[6])
redis.call('XDEL', KEYS[3], ARGV[6])
redis.call('DEL', KEYS[1])
redis.call('HDEL', KEYS[7], ARGV[7])
redis.call('HSET', KEYS[8], ARGV[7], final['sequence'] or 0)
return canceled and 2 or 1
