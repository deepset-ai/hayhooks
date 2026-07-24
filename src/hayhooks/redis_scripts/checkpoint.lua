-- hayhooks:checkpoint
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return 0 end
local current_payload = redis.call('GET', KEYS[2])
if not current_payload then return 0 end
local current = cjson.decode(current_payload)
local candidate = cjson.decode(ARGV[2])
local canceled = current['cancel_requested_at'] ~= nil and current['cancel_requested_at'] ~= cjson.null
if canceled then
    candidate['cancel_requested_at'] = current['cancel_requested_at']
    candidate['cancel_reason'] = current['cancel_reason']
    candidate['bounded_progress'] = current['bounded_progress']
    candidate['sequence'] = math.max(candidate['sequence'] or 0, current['sequence'] or 0)
end
redis.call('SET', KEYS[2], cjson.encode(candidate))
local old_status = current['status']
local new_status = candidate['status']
if new_status == 'completed' or new_status == 'failed' or new_status == 'canceled' then
    redis.call('HDEL', KEYS[4], ARGV[3])
else
    redis.call('HSET', KEYS[4], ARGV[3], candidate['definition_revision'])
end
redis.call('HSET', KEYS[5], ARGV[3], candidate['sequence'] or 0)
if old_status ~= new_status then
    local remaining = redis.call('HINCRBY', KEYS[3], old_status, -1)
    if remaining < 0 then redis.call('HSET', KEYS[3], old_status, 0) end
    redis.call('HINCRBY', KEYS[3], new_status, 1)
end
return canceled and 2 or 1
