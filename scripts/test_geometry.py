import math

from geometry import Geometry


print("\n==============================")
print("RouteTREE Geometry Test")
print("==============================")

############################################################
# Distance
############################################################

p1 = (10, 20)
p2 = (40, 60)

distance = Geometry.distance(p1, p2)

print("\nDistance")
print("------------------------------")
print("P1 :", p1)
print("P2 :", p2)
print("Distance :", round(distance, 3))

############################################################
# Midpoint
############################################################

mid = Geometry.midpoint(p1, p2)

print("\nMidpoint")
print("------------------------------")
print(mid)

############################################################
# Direction Vector
############################################################

vector = Geometry.direction_vector(p1, p2)

print("\nDirection Vector")
print("------------------------------")
print(vector)

############################################################
# Normalize
############################################################

unit = Geometry.normalize(vector)

print("\nNormalized Vector")
print("------------------------------")
print(unit)

############################################################
# Angle
############################################################

a = (0, 0)
b = (5, 0)
c = (5, 5)

angle = Geometry.angle(a, b, c)

print("\nAngle")
print("------------------------------")
print("Expected ≈ 90°")
print("Computed :", round(angle, 2))

############################################################
# Straight Line
############################################################

a = (0, 0)
b = (5, 0)
c = (10, 0)

angle = Geometry.angle(a, b, c)

print("\nStraight Line Angle")
print("------------------------------")
print("Expected ≈ 180°")
print("Computed :", round(angle, 2))

############################################################
# Turn Detection
############################################################

turn = Geometry.is_turn(
    (0, 0),
    (5, 0),
    (5, 5)
)

print("\nTurn Detection")
print("------------------------------")
print(turn)

############################################################
# Point-Line Distance
############################################################

point = (5, 5)

line_start = (0, 0)

line_end = (10, 0)

dist = Geometry.point_line_distance(
    point,
    line_start,
    line_end
)

print("\nPoint-Line Distance")
print("------------------------------")
print(round(dist, 3))

############################################################
# Centroid
############################################################

points = [

    (10, 10),

    (12, 10),

    (11, 12),

    (13, 11)

]

center = Geometry.centroid(points)

print("\nCentroid")
print("------------------------------")
print(center)

############################################################
# Merge Close Points
############################################################

points = [

    (10, 10),

    (11, 11),

    (12, 10),

    (80, 80),

    (81, 79),

    (200, 200)

]

merged = Geometry.merge_close_points(
    points,
    distance_threshold=5
)

print("\nMerged Points")
print("------------------------------")

for p in merged:

    print(p)

############################################################

print("\n==============================")
print("Geometry Test Completed")
print("==============================")