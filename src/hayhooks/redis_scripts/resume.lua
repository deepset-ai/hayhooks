-- hayhooks:resume
local previous = redis.call('GET', KEYS[1])
if not previous then return 0 end
local decoded = cjson.decode(previous)
if decoded['status'] ~= 'waiting' then return 0 end
local candidate = cjson.decode(ARGV[1])
redis.call('SET', KEYS[1], ARGV[1])
redis.call('XADD', KEYS[2], '*', 'execution_id', ARGV[2])
local remaining = redis.call('HINCRBY', KEYS[3], 'waiting', -1)
if remaining < 0 then redis.call('HSET', KEYS[3], 'waiting', 0) end
redis.call('HINCRBY', KEYS[3], candidate['status'], 1)
redis.call('HSET', KEYS[4], ARGV[2], candidate['definition_revision'])
redis.call('HSET', KEYS[5], ARGV[2], candidate['sequence'] or 0)
return 1
