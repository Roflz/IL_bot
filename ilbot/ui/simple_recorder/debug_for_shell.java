// Welcome to the RuneLite Development Shell
// Everything executed here runs on the client thread by default.
// By default client, clientThread, configManager and log are in scope
// You can subscribe to the Event Bus by using subscribe(Event.class, ev -> handler);
// and you can access things in the global injector module with var thing = inject(Thing.class);
// Press Ctrl+R or F10 to execute the contents of this editor

log.info("Hello {}", client.getGameState());

import java.util.HashSet;
import java.util.Set;

private static boolean isSimple(Class<?> t)
{
    return t.isPrimitive()
        || t == String.class
        || Number.class.isAssignableFrom(t)
        || t == Boolean.class
        || t.isEnum();
}

private static boolean isObf(String name)
{
    return name != null && name.matches("[a-z]{1,2}");
}

public void discover(Object obj)
{
    if (obj == null) {
        log.info("Object is null");
        return;
    }

    Class<?> clazz = obj.getClass();
    log.info("Class: {}", clazz.getName());

    // fields
    for (var field : clazz.getFields())
    {
        if (isObf(field.getName())) continue; // <<< hide 1–2 letter names

        field.setAccessible(true);
        try {
            Object v = field.get(obj);
            String val =
                (v == null) ? "null" :
                isSimple(field.getType()) ? String.valueOf(v) :
                "<" + field.getType().getSimpleName() + ">";
            log.info("  Field: {} : {} = {}", field.getName(), field.getType().getSimpleName(), val);
        } catch (Throwable t) {
            log.info("  Field: {} (error: {})", field.getName(), t.toString());
        }
    }

    // methods (only list signatures; *safely* invoke getters/is* returning simple types)
    for (var m : clazz.getMethods())
    {
        if (isObf(m.getName())) continue; // <<< hide 1–2 letter names

        m.setAccessible(true);
        String sig = m.getName() + "(" + m.getParameterCount() + " params) : " + m.getReturnType().getSimpleName();

        if (m.getParameterCount() == 0 &&
            (m.getName().startsWith("get") || m.getName().startsWith("is")))
        {
            try {
                Object r = m.invoke(obj);
                log.info("  Method: {} -> {} ({})", m.getName() + "()", r,
                    (r != null ? r.getClass().getSimpleName() : "null"));
            } catch (Throwable t) {
                log.info("  Method: {} (error: {})", m.getName() + "()", t.toString());
            }
        }
        else
        {
            log.info("  Method: {}", sig);
        }
    }
}

import net.runelite.api.coords.WorldPoint;
import net.runelite.api.coords.LocalPoint;
import net.runelite.api.Perspective;
import net.runelite.api.Point;
import net.runelite.api.Tile;

WorldPoint wp = client.getLocalPlayer().getWorldLocation();
LocalPoint lp = LocalPoint.fromWorld(client, wp);
Point canvas = (lp != null) ? Perspective.localToCanvas(client, lp, client.getPlane()) : null;

log.info("Player tile {} -> canvas {}", wp, canvas);
log.info("lp: {}", wp);
log.info("wp: {}", lp);
log.info("canvas: {}", canvas);

//Player localPlayer = client.getLocalPlayer();
//WorldPoint playerPos = localPlayer.getWorldLocation();
//discover(localPlayer);
//discover(playerPos);

WorldPoint here = client.getLocalPlayer().getWorldLocation();
int plane = client.getPlane();

for (int dx = -5; dx <= 5; dx++)
{
    for (int dy = -5; dy <= 5; dy++)
    {
        WorldPoint wp = new WorldPoint(here.getX() + dx, here.getY() + dy, plane);
        LocalPoint lp = LocalPoint.fromWorld(client, wp);
        if (lp == null) continue;

        Point canvas = Perspective.localToCanvas(client, lp, plane);
        log.info("Tile {} -> canvas {}", wp, canvas);
    }
}

WorldPoint here = client.getLocalPlayer().getWorldLocation();
int plane = client.getPlane();
net.runelite.api.Tile[][][] tiles = client.getScene().getTiles();

for (int dx = -2; dx <= 2; dx++) {
    for (int dy = -2; dy <= 2; dy++) {
        WorldPoint wp = new WorldPoint(here.getX() + dx, here.getY() + dy, plane);
        LocalPoint lp = LocalPoint.fromWorld(client, wp);
        if (lp == null) continue;

        Point canvas = Perspective.localToCanvas(client, lp, plane);
        int sx = lp.getSceneX(), sy = lp.getSceneY();
        if (sx < 0 || sy < 0 || sx >= 104 || sy >= 104) continue;

        Tile tile = tiles[plane][sx][sy];
        if (tile == null) continue;
//        discover(tile);

        log.info("Tile {} -> canvas {}", wp, canvas);

        net.runelite.api.WallObject wall = tile.getWallObject();
        if (wall != null) {
            int id = wall.getId();
            if (id > 0) log.info("  WallObject id={} name={}", id, client.getObjectDefinition(id).getName());
        }

        net.runelite.api.GroundObject ground = tile.getGroundObject();
        if (ground != null) {
            int id = ground.getId();
            if (id > 0) log.info("  GroundObject id={} name={}", id, client.getObjectDefinition(id).getName());
        }

        net.runelite.api.GameObject[] gos = tile.getGameObjects();
        if (gos != null) {
            for (int i = 0; i < gos.length; i++) {
                net.runelite.api.GameObject go = gos[i];
                if (go == null) continue;
                int id = go.getId();
                if (id > 0) log.info("  GameObject id={} name={}", id, client.getObjectDefinition(id).getName());
            }
        }
    }
}

